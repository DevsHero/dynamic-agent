use crate::config::prompt::{ self, PromptConfig };
use crate::llm::chat::ChatClient;
use crate::llm::embedding::EmbeddingClient;

use log::info;
use serde::{ Deserialize, Serialize };
use serde_json::Value;
use vector_nexus::schema::IndexSchema;
use vector_nexus::VectorStore;

use std::{ error::Error as StdError, sync::Arc };
use std::fmt;
use strsim;

#[derive(Debug, Deserialize)]
pub struct RagQueryArgs {
    pub query: String,
    pub limit: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoredDocument {
    pub score: f32,
    pub document: Value,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct RagQueryOutput {
    pub results: Vec<ScoredDocument>,
    pub message: String,
}

#[derive(Debug)]
pub struct RagEngineError(pub String);

impl fmt::Display for RagEngineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RagEngine Error: {}", self.0)
    }
}

impl StdError for RagEngineError {}

#[derive(Clone)]
pub struct RagEngine {
    vector_store: Arc<dyn VectorStore>,
    chat_client: Arc<dyn ChatClient>,
    embedding_client: Arc<dyn EmbeddingClient>,
    _query_generation_client: Arc<dyn ChatClient>,
    index_schemas: Vec<IndexSchema>,
    prompt_config: Arc<PromptConfig>,
    _function_schema: Value,
    _vector_type: String,
    rag_default_limit: usize,
    use_llm_query: bool,
}

impl RagEngine {
    pub fn new(
        vector_store: Arc<dyn VectorStore>,
        chat_client: Arc<dyn ChatClient>,
        embedding_client: Arc<dyn EmbeddingClient>,
        _query_generation_client: Arc<dyn ChatClient>,
        index_schemas: Vec<IndexSchema>,
        prompt_config: Arc<PromptConfig>,
        _function_schema: Value,
        _vector_type: String,
        rag_default_limit: usize,
        use_llm_query: bool
    ) -> Self {
        Self {
            vector_store,
            chat_client,
            embedding_client,
            _query_generation_client,
            index_schemas,
            prompt_config,
            _function_schema,
            _vector_type,
            rag_default_limit,
            use_llm_query,
        }
    }

    fn format_documents_for_prompt(hits: &Vec<(f32, String, Value)>) -> String {
        if hits.is_empty() {
            return "No relevant documents found.".to_string();
        }

        let mut docs_text = String::new();
        for (score, id, doc_value) in hits {
            docs_text.push_str(&format!("Document ID: {} (Score: {:.4})\n", id, score));
            if let Some(doc_obj) = doc_value.as_object() {
                if doc_obj.is_empty() {
                    docs_text.push_str("  - (No fields retrieved for this document)\n");
                } else {
                    for (key, value) in doc_obj {
                        if
                            key == "vector" ||
                            key == "pdf" ||
                            key == "describe_pdf_data" ||
                            key == "portfolio_detail_pdf_data"
                        {
                            continue;
                        }
                        let value_str = match value {
                            Value::String(s) => s.clone(),
                            _ => value.to_string(),
                        };
                        docs_text.push_str(&format!("  - {}: {}\n", key, value_str));
                    }
                }
            } else {
                docs_text.push_str("  - Document content is not a valid JSON object.\n");
            }
            docs_text.push('\n');
        }
        docs_text
    }

    pub async fn query_and_answer(
        &self,
        args: RagQueryArgs,
        user_question: &str
    ) -> Result<String, Box<dyn StdError + Send + Sync>> {
        let schema_json_for_inference = serde_json::to_string(&self.index_schemas)?;
        let topic_inference_prompt = prompt::get_rag_topic_prompt(
            &self.prompt_config,
            &schema_json_for_inference,
            user_question
        )?;
        
        info!("--- Topic Inference Prompt ---\n{}\n-----------------------------", topic_inference_prompt);
        
        let topic_resp = self.chat_client.complete(&topic_inference_prompt).await?;
        let inferred_topic = topic_resp.response.trim().trim_matches('"').to_lowercase();
        
        info!("--- Inferred Topic (Trimmed, No Quotes, Lowercased): '{}' ---", inferred_topic);
        
        let final_topic = if inferred_topic.is_empty() || 
                           inferred_topic == "none" || 
                           !self.index_schemas.iter().any(|s| s.name == inferred_topic) {
            info!("Primary topic inference failed, trying fallback resolver");
            
            let schema_summary = self.index_schemas.iter()
                .map(|s| format!("- {}: fields={}", s.name, s.fields.join(", ")))
                .collect::<Vec<_>>()
                .join("\n");
                
            let fallback_prompt = prompt::get_fallback_topic_prompt(
                &self.prompt_config,
                &schema_summary,
                user_question
            )?;
            
            let fallback_resp = self.chat_client.complete(&fallback_prompt).await?;
            let fallback_topic = fallback_resp.response.trim().trim_matches('"').to_lowercase();
            
            info!("--- Fallback Topic Resolution: '{}' ---", fallback_topic);
            
            if fallback_topic.is_empty() || 
               fallback_topic == "none" || 
               !self.index_schemas.iter().any(|s| s.name == fallback_topic) {
                return Err(Box::new(RagEngineError(
                    "Could not determine the correct data category for your question after multiple attempts. Please try rephrasing.".into()
                )));
            }
            
            fallback_topic
        } else {
            inferred_topic
        };
        
        let lower_q = user_question.to_lowercase();
        if
            (lower_q.contains("count") ||
                lower_q.contains("total") ||
                lower_q.contains("how many") ||
                lower_q.contains("how much")) &&
            !final_topic.is_empty()
        {
            let cnt = self.vector_store
                .count_documents(&final_topic).await
                .map_err(|e| Box::new(RagEngineError(format!("Count failed: {}", e))))?;
            return Ok(cnt.to_string());
        }

        let embed_resp = self.embedding_client
            .embed(&args.query).await
            .map_err(|e| Box::new(RagEngineError(format!("Embedding failed: {}", e))))?;
        let vec_f32 = embed_resp.embedding;
        let available_fields = self.index_schemas
            .iter()
            .find(|s| s.name == final_topic)
            .map(|s| s.fields.as_slice())
            .unwrap_or(&[]);

        let selected_fields = if self.use_llm_query {
            info!("→ Attempting to resolve dynamic fields using LLM...");
            self.resolve_dynamic_fields(user_question, available_fields).unwrap_or_else(|| {
                info!(
                    "→ LLM field resolution failed or returned none, falling back to all fields."
                );
                available_fields.to_vec()
            })
        } else {
            info!("→ Skipping LLM field resolution, using all available fields.");
            available_fields.to_vec()
        };

        let mut hits = {
            let limit = args.limit.unwrap_or(self.rag_default_limit);

            info!("→ Performing search with selected fields: {:?}", selected_fields);

            self.vector_store.search_hybrid(
                &final_topic,
                user_question,
                &vec_f32,
                limit,
                Some(&selected_fields)
            ).await?
        };

        if
            (final_topic == "experience" ||
                final_topic == "education" ||
                final_topic == "portfolio") &&
            (lower_q.contains("latest") ||
                lower_q.contains("recent") ||
                lower_q.contains("newest") ||
                lower_q.contains("current"))
        {
            hits.sort_by(|(_, _, a), (_, _, b)| {
                let a_date = a
                    .get("end_date")
                    .and_then(|d| d.as_str())
                    .unwrap_or("0000-00-00");

                let b_date = b
                    .get("end_date")
                    .and_then(|d| d.as_str())
                    .unwrap_or("0000-00-00");

                b_date.cmp(a_date)
            });

            if hits.len() > 1 {
                hits = vec![hits[0].clone()];
                info!("Filtered to latest entry by end_date");
            }
        }

        let docs_text = Self::format_documents_for_prompt(&hits);

        let retrieved_topics = if hits.is_empty() {
            "none".to_string()
        } else {
            final_topic.clone()
        };

        let schema_json_for_answer = serde_json
            ::to_string_pretty(&self.index_schemas)
            .map_err(|e| Box::new(RagEngineError(format!("Schema JSON error for answer: {}", e))))?;

        let final_prompt = prompt::get_rag_final_prompt(
            &self.prompt_config,
            &schema_json_for_answer,
            &retrieved_topics,
            &docs_text,
            user_question
        )?;

        info!("--- Final Answer Prompt ---\n{}\n--------------------------", final_prompt);

        let answer_resp = self.chat_client
            .complete(&final_prompt).await
            .map_err(|e| Box::new(RagEngineError(format!("Final completion failed: {}", e))))?;

        Ok(answer_resp.response)
    }

    fn resolve_dynamic_fields(
        &self,
        user_question: &str,
        available_fields: &[String]
    ) -> Option<Vec<String>> {
        let mut q = user_question
            .trim_end_matches(|c: char| c.is_ascii_punctuation())
            .to_lowercase();

        if let Some(pos) = q.find(" from ") {
            q.truncate(pos);
        }

        let mut words: Vec<&str> = q.split_whitespace().collect();
        if let Some(first) = words.first() {
            let verbs = ["list", "show", "give", "tell", "what", "find"];
            if verbs.contains(first) {
                words.remove(0);
            }
        }

        let term = words.last().unwrap_or(&"");
        let norm_term = term.replace(&['_', '-', ' '][..], "");

        for f in available_fields {
            if f.to_lowercase().replace('_', "") == norm_term {
                return Some(vec![f.clone()]);
            }
        }

        let mut best: Option<&String> = None;
        let mut best_score = 0.0;
        for f in available_fields {
            let candidate = f.to_lowercase().replace('_', "");
            let score = strsim::jaro_winkler(&norm_term, &candidate);
            if score > best_score {
                best_score = score;
                best = Some(f);
            }
        }
        if best_score >= 0.85 {
            return best.map(|f| vec![f.clone()]);
        }

        None
    }
}
