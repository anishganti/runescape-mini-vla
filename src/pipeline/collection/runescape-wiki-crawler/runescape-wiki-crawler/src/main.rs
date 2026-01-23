mod structs;

use structs::{ScrapeResponse, ListResponse, ProcessedPage, Query, Page, Continuation};
use anyhow::Result;
use serde::Deserialize;
use reqwest::Client;
use serde_json::Value;
use tokio::task;
use std::time::Duration;
use tokio::time::interval;
use std::collections::HashSet;
use tokio::fs::OpenOptions;
use tokio::io::AsyncWriteExt; // Required for write_all and set_lenuse std::io::Write;
use std::sync::{Arc, Mutex};


fn param(k: &str, v: impl Into<String>) -> (String, String) {
    (k.to_string(), v.into())
}

fn join_titles(titles: &Vec<String>) -> (String) {
    ("".to_string())
}

async fn call_osrs_wiki_api<T>(
    client: &Client, 
    function: String, 
    cont: Option<Continuation>, 
    ttls: Option<String>
) -> Result<T>
where T: serde::de::DeserializeOwned 
{

    let base_url = "https://oldschool.runescape.wiki/api.php";

    let mut params: Vec<(String, String)> = vec![
        param("action", "query"),
        param("format", "json"),
    ];

    if function == "list".to_string() {
        params.push(param("list", "allpages"));   
        params.push(param("aplimit", "500"));
    } else if function == "scrape".to_string() {
        params.push(param("prop", "revisions"));
        params.push(param("rvprop", "content"));
        params.push(param("rvslots", "main"));
        params.push(param("redirects", "1"));      
    }

    if let Some(continuation) = cont {
        params.push(param("apcontinue", continuation.apcontinue));
        params.push(param("continue", continuation.continuation));    
    }

    if let Some(titles) = ttls {
        params.push(param("titles", titles))
    }

    let response = client.get(base_url).query(&params).send().await?;
    let status = response.status();
    let body_text = response.text().await?;

    if !status.is_success() {
        return Err(anyhow::anyhow!("HTTP {}: {}", status, body_text));
    }

    // The result of this MATCH is the return value of the function
    match serde_json::from_str::<T>(&body_text) {
        Ok(parsed) => Ok(parsed), 
        Err(e) => {
            eprintln!("JSON Error: {}", e);
            eprintln!("Body: {}", body_text);
            Err(e.into()) // This now correctly flows into the function's return type
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    /*
        Why create reqwest::Client instead of reqwest::get? While the latter is fine for a
        one-time HTTP request, it has to reinitialize the client under the hood. Since we 
        are making ~100K requests, it's better to just create a one-time Client that can be 
        reused across all HTTP requests. 
     */
    let (tx, rx) = flume::bounded::<String>(5000);
    let (writer_tx, writer_rx) = flume::bounded::<ProcessedPage>(5000);

    let mut file = OpenOptions::new()
    .create(true)
    .append(true)
    .open("wiki_data.jsonl")
    .await?; // Notice the .await here

    let client = Client::builder()
    .user_agent("learning-crawler/0.1 (contact: anishgantis@utexas.edu)")
    .build()?;

    let producer_tx = tx.clone();
    let client_tx = client.clone();
    
    let producer = task::spawn(async move {
        let mut next_continue: Option<Continuation> = None;

        loop {
            let resp_result = call_osrs_wiki_api::<ListResponse>(
                &client_tx, 
                "list".to_string(), 
                next_continue.clone(), 
                None
            ).await;

            match resp_result {
                Ok(resp) => {
                    for page in resp.query.allpages {
                        producer_tx.send_async(page.title).await.unwrap();
                    }
        
                    if resp.continuation.is_some() {
                        next_continue = resp.continuation; 
                    } else {
                        println!("Producer: Reached the end of the wiki.");
                        break;
                    }
                }
                Err(e) => {
                    eprintln!("API Error: {}", e);
                    // To see the actual HTML/Text causing the JSON error, 
                    // you'd need to modify call_osrs_wiki_api to return the raw text on error.
                    break; 
                }
            }
        }
    });

    let mut consumers = vec![];
    for _i in 0..5 {
        let consumer_rx = rx.clone();
        let writer_tx = writer_tx.clone();
        let client_rx = client.clone();

        let consumer = task::spawn(async move {
            let mut batch = Vec::with_capacity(50);
            let mut timer = interval(Duration::from_millis(500));

            loop {
                let mut scrape = false;
                tokio::select! {
                    page = consumer_rx.recv_async() => {
                        match page {
                            Ok(p) => {
                                batch.push(p);
                                if batch.len() >= 5 {
                                    scrape = true;
                                }
                            }
                            Err(_) => break, 
                        }
                    }
                    _ = timer.tick() => {
                        if !batch.is_empty() {
                            scrape = true;
                        }
                    }
                }

                if scrape {
                    let titles_raw = std::mem::take(&mut batch);
                    let titles_query = titles_raw.join("|");
                    
                    if let Ok(response) = call_osrs_wiki_api::<ScrapeResponse>(&client_rx, "scrape".to_string(), None, Some(titles_query)).await {
                        let processed = response.flatten();
                        
                        for page in processed {
                            let _ = writer_tx.send_async(page).await;
                        }
                    }            
                }
            }
        });

        consumers.push(consumer);
    }

    let writer = task::spawn_blocking(move || {
        let mut canonical_page_ids = HashSet::new();
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open("wiki_data.jsonl")
            .expect("Could not open file");
    
        while let Ok(page) = writer_rx.recv() {
            if canonical_page_ids.insert(page.page_id) {
                if let Ok(line) = serde_json::to_string(&page) {
                    use std::io::Write;
                    writeln!(file, "{}", line).unwrap();
                }
            }
        }
        println!("Writer task: Channel closed. Finishing up...");
    });

    drop(tx); 
    drop(writer_tx);
    
    let _ = producer.await;

    for consumer in consumers {
        let _ = consumer.await;
    }
    
    let _ = writer.await;

    Ok(())
}