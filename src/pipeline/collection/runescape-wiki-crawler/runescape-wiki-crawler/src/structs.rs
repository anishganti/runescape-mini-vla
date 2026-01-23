use serde::{Deserialize, Serialize}; // Add Serialize here
use std::collections::HashMap;

#[derive(Debug, Deserialize)]
pub struct ListResponse {
    pub batchcomplete: String,
    #[serde(rename = "continue")]
    pub continuation: Option<Continuation>, 
    pub query: Query                
}

#[derive(Debug, Deserialize, Clone)]
pub struct Continuation {
    pub apcontinue: String,
    #[serde(rename = "continue")]
    pub continuation: String
}

#[derive(Debug, Deserialize)]
pub struct Query{
    pub allpages: Vec<Page>
}

#[derive(Debug, Deserialize)]
pub struct Page{
    pub ns: i8, 
    pub pageid: i32, 
    pub title: String
}
//------------------ Scrape Page Response Struct ----------//
#[derive(Deserialize, Debug)]
pub struct ScrapeResponse {
    pub query: ScrapeQuery,
}

#[derive(Deserialize, Debug)]
pub struct ScrapeQuery {
    // This handles the dynamic "40681" keys
    pub pages: HashMap<String, ScrapePage>,
    pub redirects: Option<Vec<Redirect>>,
}

#[derive(Deserialize, Debug)]
pub struct ScrapePage {
    pub pageid: u32,
    pub title: String,
    pub revisions: Vec<Revision>,
}

#[derive(Deserialize, Debug)]
pub struct Revision {
    pub slots: Slots,
}

#[derive(Deserialize, Debug)]
pub struct Slots {
    pub main: MainSlot,
}

#[derive(Deserialize, Debug)]
pub struct MainSlot {
    #[serde(rename = "*")]
    pub content: String,
    pub contentformat: String,
    pub contentmodel: String,
}

#[derive(Deserialize, Debug)]
pub struct Redirect {
    pub from: String,
    pub to: String,
    pub tofragment: Option<String>,
}

// --- THE "WORKER" STRUCT (This is what you use) ---
pub struct FlattenedPage {
    pub id: u32,
    pub title: String,
    pub content: String,
    pub redirects: Vec<(String, String)>, // (From, To)
}

/// --- 2. THE FINAL CLEAN DATA STRUCT ---

#[derive(Debug, Serialize)]
pub struct ProcessedPage {
    pub page_id: u32,
    pub title: String,
    pub main_content: String,
    pub redirected_from: Vec<String>,
}

// --- 3. THE FLATTEN LOGIC ---

impl ScrapeResponse {
    /// Converts the nested API mess into a simple List of ProcessedPages
    pub fn flatten(self) -> Vec<ProcessedPage> {
        let all_redirects = self.query.redirects.unwrap_or_default();

        self.query.pages.into_values().map(|page| {
            // Safely grab content from the first revision
            let content = page.revisions
                .into_iter()
                .next()
                .map(|rev| rev.slots.main.content)
                .unwrap_or_else(|| "".to_string());

            // Collect any "from" redirects that point to this page's title
            let from_list = all_redirects.iter()
                .filter(|r| r.to == page.title)
                .map(|r| r.from.clone())
                .collect();

            ProcessedPage {
                page_id: page.pageid,
                title: page.title,
                main_content: content,
                redirected_from: from_list,
            }
        }).collect()
    }
}