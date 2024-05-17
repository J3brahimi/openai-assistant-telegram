use async_openai::{
    types::{
        CreateMessageRequestArgs, CreateRunRequestArgs, CreateThreadRequestArgs, MessageContent,
        RunStatus,
    },
    Client,
};
use flowsnet_platform_sdk::logger;
use tg_flows::{listen_to_update, update_handler, Telegram, UpdateKind};
use reqwest::Client as ReqwestClient;
use std::env;
use tokio::fs::File;
use tokio::io::AsyncWriteExt;

#[no_mangle]
#[tokio::main(flavor = "current_thread")]
pub async fn on_deploy() {
    logger::init();

    let telegram_token = env::var("telegram_token").unwrap();
    listen_to_update(telegram_token).await;
}

#[update_handler]
async fn handler(update: tg_flows::Update) {
    logger::init();
    let telegram_token = env::var("telegram_token").unwrap();
    let tele = Telegram::new(telegram_token);

    if let UpdateKind::Message(msg) = update.kind {
        let chat_id = msg.chat.id;

        if let Some(voice) = msg.voice() {
            let file_id = voice.file.file_id.clone();
            if let Some(file_path) = download_voice_file(&tele, &file_id).await {
                let text = transcribe_audio(file_path).await;
                let response = run_message(&text).await;
                _ = tele.send_message(chat_id, response).await;
            }
        } else if let Some(text) = msg.text() {
            let thread_id = match store_flows::get(chat_id.to_string().as_str()) {
                Some(ti) => {
                    if text == "/restart" {
                        delete_thread(ti.as_str().unwrap()).await;
                        store_flows::del(chat_id.to_string().as_str());
                        return;
                    } else {
                        ti.as_str().unwrap().to_owned()
                    }
                }
                None => {
                    let ti = create_thread().await;
                    store_flows::set(
                        chat_id.to_string().as_str(),
                        serde_json::Value::String(ti.clone()),
                        None,
                    );
                    ti
                }
            };

            let response = run_message(text).await;
            _ = tele.send_message(chat_id, response).await;
        }
    }
}

async fn download_voice_file(tele: &Telegram, file_id: &str) -> Option<String> {
    let file_info = tele.get_file(file_id.to_string()).ok()?;
    let file_path = format!("https://api.telegram.org/file/bot{}/{}", env::var("telegram_token").unwrap(), file_info.file_path);
    let file_name = format!("/tmp/{}.ogg", file_id);

    let client = ReqwestClient::new();
    let mut response = client.get(&file_path).send().await.ok()?;
    let mut file = File::create(&file_name).await.ok()?;
    while let Some(chunk) = response.chunk().await.ok()? {
        file.write_all(&chunk).await.ok()?;
    }

    Some(file_name)
}

async fn transcribe_audio(file_path: String) -> String {
    let client = ReqwestClient::new();
    let file = tokio::fs::read(file_path.clone()).await.unwrap();

    let form = reqwest::multipart::Form::new()
        .file("file", &file_path)
        .unwrap()
        .text("model", "whisper-1");

    let response = client
        .post("https://api.openai.com/v1/audio/transcriptions")
        .header("Authorization", format!("Bearer {}", env::var("OPENAI_API_KEY").unwrap()))
        .multipart(form)
        .send()
        .await
        .unwrap();

    let transcription: serde_json::Value = response.json().await.unwrap();
    transcription["text"].as_str().unwrap().to_string()
}

async fn create_thread() -> String {
    let client = Client::new();

    let create_thread_request = CreateThreadRequestArgs::default().build().unwrap();

    match client.threads().create(create_thread_request).await {
        Ok(to) => {
            log::info!("New thread (ID: {}) created.", to.id);
            to.id
        }
        Err(e) => {
            panic!("Failed to create thread. {:?}", e);
        }
    }
}

async fn delete_thread(thread_id: &str) {
    let client = Client::new();

    match client.threads().delete(thread_id).await {
        Ok(_) => {
            log::info!("Old thread (ID: {}) deleted.", thread_id);
        }
        Err(e) => {
            log::error!("Failed to delete thread. {:?}", e);
        }
    }
}

async fn run_message(text: &str) -> String {
    let client = Client::new();
    let assistant_id = env::var("ASSISTANT_ID").unwrap();
    let thread_id = create_thread().await;

    let create_message_request = CreateMessageRequestArgs::default()
        .content(text.to_string())
        .build()
        .unwrap();

    client
        .threads()
        .messages(&thread_id)
        .create(create_message_request)
        .await
        .unwrap();

    let create_run_request = CreateRunRequestArgs::default()
        .assistant_id(assistant_id)
        .build()
        .unwrap();
    let run_id = client
        .threads()
        .runs(&thread_id)
        .create(create_run_request)
        .await
        .unwrap()
        .id;

    let mut result = Some("Timeout");
    for _ in 0..5 {
        tokio::time::sleep(std::time::Duration::from_secs(8)).await;
        let run_object = client
            .threads()
            .runs(&thread_id)
            .retrieve(run_id.as_str())
            .await
            .unwrap();
        result = match run_object.status {
            RunStatus::Queued | RunStatus::InProgress | RunStatus::Cancelling => {
                continue;
            }
            RunStatus::RequiresAction => Some("Action required for OpenAI assistant"),
            RunStatus::Cancelled => Some("Run is cancelled"),
            RunStatus::Failed => Some("Run is failed"),
            RunStatus::Expired => Some("Run is expired"),
            RunStatus::Completed => None,
        };
        break;
    }

    match result {
        Some(r) => r.to_string(),
        None => {
            let mut thread_messages = client
                .threads()
                .messages(&thread_id)
                .list(&[("limit", "1")])
                .await
                .unwrap();

            let c = thread_messages.data.pop().unwrap();
            let c = c.content.into_iter().filter_map(|x| match x {
                MessageContent::Text(t) => Some(t.text.value),
                _ => None,
            });

            c.collect()
        }
    }
}