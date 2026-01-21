use anyhow::Result;
use serde::Deserialize;
use reqwest::Client;

#[derive(Debug, Deserialize)]
struct ApiResponse {
    batchcomplete: String,
    #[serde(rename = "continue")]
    continuation: Continuation, 
    query: Query                
}

#[derive(Debug, Deserialize)]
struct Continuation {
    apcontinue: String,
    #[serde(rename = "continue")]
    continuation: String
}

#[derive(Debug, Deserialize)]
struct Query{
    allpages: Vec<Page>
}

#[derive(Debug, Deserialize)]
struct Page{
    ns: i8, 
    pageid: i32, 
    title: String
}

fn param(k: &str, v: impl Into<String>) -> (String, String) {
    (k.to_string(), v.into())
}

async fn call_osrs_wiki_api(
    client: &Client, 
    cont: Option<Continuation>,
) -> Result<ApiResponse, Box<dyn std::error::Error>> {

    let base_url = "https://oldschool.runescape.wiki/api.php";

    let mut params: Vec<(String, String)> = vec![
        param("action", "query"),
        param("format", "json"),
        param("list", "allpages"),
        param("aplimit", "10")
    ];

    if let Some(continuation) = cont {
        params.push(param("apcontinue", continuation.apcontinue));
        params.push(param("continue", continuation.continuation));    
    }

    let resp: ApiResponse = client.get(base_url)
        .query(&params)
        .send()
        .await?
        .json()
        .await?;

    Ok(resp)
}

async fn resolve_canonical_pages() -> Result<()> {
    Ok(())
}

async fn enqueue_canonical_pages() -> Result<()> {
    Ok(())
}

async fn scrape_canonical_pages() -> Result<()> {
    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    /*
    Why create reqwest::Client instead of reqwest::get? While the latter is fine for a
    one-time HTTP request, it has to reinitialize the client under the hood. Since we 
    are making ~100K requests, it's better to just create a one-time Client that can be 
    reused across all HTTP requests. 
     */

    let client = Client::builder()
        .user_agent("learning-crawler/0.1 (contact: anishgantis@utexas.edu)")
        .build()?;

    let resp = call_osrs_wiki_api(&client, None).await;

    println!("{:#?}", resp);

    Ok(())
} 