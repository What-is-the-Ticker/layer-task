#[allow(warnings)]
mod bindings;

use anyhow::anyhow;
use bindings::{Guest, Output, TaskQueueInput};
use serde::{Deserialize, Serialize};

use markov::Chain;
use rand::seq::SliceRandom;
use rand::thread_rng;

use rand::prelude::*;
use regex::Regex;
use std::collections::HashMap;


/// The corpus is included at compile time.
/// Replace this with a larger corpus for better results.
mod constants;
use crate::constants::CORPUS;




/// The NGramModel struct holds the n-gram models.
struct NGramModel {
    models: HashMap<usize, HashMap<Vec<String>, HashMap<String, usize>>>,
    max_n: usize,
}

impl NGramModel {
    /// Constructs a new NGramModel with the given corpus and maximum n-gram size.
    fn new(corpus: &str, max_n: usize) -> Self {
        let models = NGramModel::build_ngram_model(corpus, max_n);
        NGramModel { models, max_n }
    }

    /// Builds n-gram models from the corpus.
    fn build_ngram_model(
        corpus: &str,
        max_n: usize,
    ) -> HashMap<usize, HashMap<Vec<String>, HashMap<String, usize>>> {
        let mut models: HashMap<usize, HashMap<Vec<String>, HashMap<String, usize>>> =
            HashMap::new();

        // Define a regex pattern for tokenization (words and punctuation)
        let re = Regex::new(r"\w+|[^\s\w]").unwrap();

        // Tokenize the corpus using regex
        let tokens: Vec<String> = re
            .find_iter(corpus)
            .map(|mat| mat.as_str().to_lowercase())
            .collect();

        // Build n-gram models with counts for n from 2 to max_n
        for n in 2..=max_n {
            let mut model: HashMap<Vec<String>, HashMap<String, usize>> = HashMap::new();
            for i in 0..tokens.len() - n + 1 {
                let key = tokens[i..i + n - 1].to_vec();
                let value = tokens[i + n - 1].clone();
                model
                    .entry(key)
                    .or_insert_with(HashMap::new)
                    .entry(value)
                    .and_modify(|count| *count += 1)
                    .or_insert(1);
            }
            models.insert(n, model);
        }

        models
    }

    /// Generates text using the n-gram models.
    fn generate_text(
        &self,
        input: &str,
        length: usize,
        temperature: f64,
        top_k: usize,
    ) -> String {
        let mut rng = thread_rng();

        // Initialize the current key
        let mut current_key: Vec<String> = if !input.trim().is_empty() {
            input
                .split_whitespace()
                .map(|s| s.to_lowercase())
                .collect()
        } else {
            // Start with a random key from the highest-order model
            let model = self.models.get(&self.max_n).unwrap();
            let keys: Vec<&Vec<String>> = model.keys().collect();
            keys.choose(&mut rng).unwrap().to_vec()
        };

        let mut result = current_key.join(" ");

        for _ in 0..length {
            let mut next_word = None;

            // Try to find the next word using decreasing n-gram sizes
            for n in (2..=self.max_n).rev() {
                if current_key.len() >= n - 1 {
                    let key_slice = current_key[current_key.len() - (n - 1)..].to_vec();
                    if let Some(model) = self.models.get(&n) {
                        if let Some(possible_next_words) = model.get(&key_slice) {
                            // Calculate probabilities with temperature
                            let mut word_probs: Vec<(String, f64)> = possible_next_words
                                .iter()
                                .map(|(word, &count)| {
                                    let prob = (count as f64).powf(1.0 / temperature);
                                    (word.clone(), prob)
                                })
                                .collect();

                            // Sort and keep top_k words
                            word_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                            word_probs.truncate(top_k);

                            // Normalize probabilities
                            let total_prob: f64 = word_probs.iter().map(|&(_, prob)| prob).sum();
                            let mut cumulative_prob = 0.0;
                            let mut probs = Vec::new();
                            for (word, prob) in &word_probs {
                                cumulative_prob += prob / total_prob;
                                probs.push((word.clone(), cumulative_prob));
                            }

                            // Sample the next word
                            let rnd: f64 = rng.gen();
                            for (word, prob) in probs {
                                if rnd <= prob {
                                    next_word = Some(word);
                                    break;
                                }
                            }
                            break;
                        }
                    }
                }
            }

            // If no next word is found, break the loop
            if let Some(word) = next_word {
                result.push(' ');
                result.push_str(&word);
                current_key.push(word);
            } else {
                break;
            }
        }

        result
    }
}



#[derive(Deserialize, Debug)]
pub struct TaskRequestData {
    pub prompt: String,
}

#[derive(Serialize, Debug)]
pub struct TaskResponseData {
    word: String,
}
struct Component;

impl Guest for Component {
    fn run_task(request: TaskQueueInput) -> Output {
        let TaskRequestData { prompt } = serde_json::from_slice(&request.request)
            .map_err(|e| anyhow!("Could not deserialize input request from JSON: {}", e))
            .unwrap();
        let max_n = 3;          // Maximum n-gram size
        let length = 2;        // Number of words to generate
        let temperature = 0.5;  // Temperature for randomness
        let top_k = 5;          // Top-K sampling
    
        // Build the n-gram model
        let model = NGramModel::new(CORPUS, max_n);
    
        // Generate text using the model
        let generated_text = model.generate_text(&prompt, length, temperature, top_k);

        //merge generated_text with corpus and create a new training data
        let training_data = format!("{}{}", CORPUS, generated_text);

        let mut chain = Chain::new();
        chain.feed_str(training_data.as_str());

        let mut rng = thread_rng();
        let generated_text = chain.generate_str();

        // Split the generated text into words
        let words: Vec<&str> = generated_text.split_whitespace().collect();

        let random_word = words.choose(&mut rng).unwrap();

        Ok(serde_json::to_vec(&TaskResponseData { word: random_word.to_string() })
            .map_err(|e| anyhow!("Could not serialize output data into JSON: {}", e))
            .unwrap())
    }
}

bindings::export!(Component with_types_in bindings);
