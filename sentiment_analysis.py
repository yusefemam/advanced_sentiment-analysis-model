#Made by Youssef Emam Linguistics Engineering (NLP Engineer) 
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel, GPT2Tokenizer, GPT2LMHeadModel, AdamW
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np
import logging
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import streamlit as st


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('nlp_model.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


CONFIG = {
    'model': {
        'bert_model_name': 'bert-base-uncased',
        'gpt_model_name': 'gpt2',
        'max_length': 512,
        'batch_size': 16,
        'learning_rate': 2e-5,
        'epochs': 3,
        'warmup_steps': 0,
        'dropout_rate': 0.1,
        'num_labels': 3
    },
    'training': {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'save_path': 'saved_models/',
        'checkpoint_frequency': 1
    },
    'generation': {
        'max_length': 100,
        'temperature': 0.7,
        'top_k': 50,
        'top_p': 0.9
    }
}

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class HybridBERTGPTModel(nn.Module):
    def __init__(self, config):
        super(HybridBERTGPTModel, self).__init__()
        
        
        self.bert = BertModel.from_pretrained(config['model']['bert_model_name'])
        self.bert_dropout = nn.Dropout(config['model']['dropout_rate'])
        
        
        self.gpt = GPT2LMHeadModel.from_pretrained(config['model']['gpt_model_name'])
        self.gpt_dropout = nn.Dropout(config['model']['dropout_rate'])
        
      
        combined_hidden_size = self.bert.config.hidden_size + self.gpt.config.n_embd
        
        
        self.classifier = nn.Sequential(
            nn.Linear(combined_hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(config['model']['dropout_rate']),
            nn.Linear(512, config['model']['num_labels'])
        )
        
        
        self.generation_head = nn.Linear(
            combined_hidden_size, 
            self.gpt.config.vocab_size
        )

    def forward(self, input_ids, attention_mask, labels=None, task='classification'):
        
        bert_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        bert_features = self.bert_dropout(bert_outputs.pooler_output)
        
        
        gpt_outputs = self.gpt(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        gpt_features = self.gpt_dropout(gpt_outputs.hidden_states[-1].mean(dim=1))
        
        
        combined_features = torch.cat((bert_features, gpt_features), dim=1)
        
        if task == 'classification':
            logits = self.classifier(combined_features)
            loss = None
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, CONFIG['model']['num_labels']), 
                              labels.view(-1))
            return loss, logits
            
        elif task == 'generation':
            generation_logits = self.generation_head(combined_features)
            return generation_logits
        
        else:
            raise ValueError(f"Unknown task: {task}")

    def generate_text(self, input_ids, attention_mask, max_length=100):
        with torch.no_grad():
            
            bert_outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            bert_features = self.bert_dropout(bert_outputs.pooler_output)
            
            
            generated = self.gpt.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                temperature=CONFIG['generation']['temperature'],
                top_k=CONFIG['generation']['top_k'],
                top_p=CONFIG['generation']['top_p'],
                num_return_sequences=1,
                pad_token_id=self.gpt.config.pad_token_id,
                do_sample=True
            )
            
            return generated

class TrainingManager:
    def __init__(self, model, config, device):
        self.model = model
        self.config = config
        self.device = device
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }

    def train_epoch(self, dataloader, optimizer, scheduler):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        progress_bar = tqdm(dataloader, desc="Training")
        
        for batch in progress_bar:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            optimizer.zero_grad()
            
            loss, logits = self.model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels']
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == batch['labels']).sum().item()
            total += batch['labels'].size(0)
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})

        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total
        
        return avg_loss, accuracy

    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0
        predictions = []
        true_labels = []

        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                loss, logits = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels']
                )
                
                total_loss += loss.item()
                predictions.extend(torch.argmax(logits, dim=1).cpu().numpy())
                true_labels.extend(batch['labels'].cpu().numpy())

        avg_loss = total_loss / len(dataloader)
        accuracy = np.mean(np.array(predictions) == np.array(true_labels))
        
        return avg_loss, accuracy, predictions, true_labels

    def train(self, train_dataloader, val_dataloader):
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.config['model']['learning_rate']
        )
        
        total_steps = len(train_dataloader) * self.config['model']['epochs']
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            total_iters=total_steps
        )

        for epoch in range(self.config['model']['epochs']):
            logger.info(f"Epoch {epoch+1}/{self.config['model']['epochs']}")
            
            train_loss, train_acc = self.train_epoch(train_dataloader, optimizer, scheduler)
            val_loss, val_acc, _, _ = self.evaluate(val_dataloader)
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            
            logger.info(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            logger.info(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            
            if (epoch + 1) % self.config['training']['checkpoint_frequency'] == 0:
                self.save_checkpoint(epoch + 1)

    def save_checkpoint(self, epoch):
        checkpoint_path = f"{self.config['training']['save_path']}checkpoint_epoch_{epoch}.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'history': self.history
        }, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")

    def plot_training_history(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        ax1.plot(self.history['train_loss'], label='Train Loss')
        ax1.plot(self.history['val_loss'], label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        
        ax2.plot(self.history['train_acc'], label='Train Accuracy')
        ax2.plot(self.history['val_acc'], label='Validation Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        
        plt.tight_layout()
        return fig

class TextAnalysisApp:
    def __init__(self):
        self.config = CONFIG
        self.setup_model()
        self.setup_tokenizers()

    def setup_model(self):
        try:
            self.device = torch.device(self.config['training']['device'])
            self.model = HybridBERTGPTModel(self.config).to(self.device)
            logger.info("Model initialized successfully")
        except Exception as e:
            logger.error(f"Error setting up model: {str(e)}")
            raise

    def setup_tokenizers(self):
        try:
            self.bert_tokenizer = BertTokenizer.from_pretrained(
                self.config['model']['bert_model_name']
            )
            self.gpt_tokenizer = GPT2Tokenizer.from_pretrained(
                self.config['model']['gpt_model_name']
            )
            self.gpt_tokenizer.pad_token = self.gpt_tokenizer.eos_token
            logger.info("Tokenizers initialized successfully")
        except Exception as e:
            logger.error(f"Error setting up tokenizers: {str(e)}")
            raise

    def train_model(self, train_texts, train_labels, val_texts, val_labels):
        try:
            
            train_dataset = TextDataset(
                train_texts, 
                train_labels, 
                self.bert_tokenizer,
                self.config['model']['max_length']
            )
            val_dataset = TextDataset(
                val_texts, 
                val_labels, 
                self.bert_tokenizer,
                self.config['model']['max_length']
            )
            
            
            train_dataloader = DataLoader(
                train_dataset,
                batch_size=self.config['model']['batch_size'],
                shuffle=True
            )
            val_dataloader = DataLoader(
                val_dataset,
                batch_size=self.config['model']['batch_size'],
                shuffle=False
            )
            
            
            training_manager = TrainingManager(self.model, self.config, self.device)
            
            
            training_manager.train(train_dataloader, val_dataloader)
            
            
            return training_manager.plot_training_history()
            
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            return None

    def analyze_text(self, text):
        try:
            inputs = self.bert_tokenizer(
                text,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=self.config['model']['max_length']
            ).to(self.device)
            
            self.model.eval()
            with torch.no_grad():
                loss, logits = self.model(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask']
                )
                prediction = torch.argmax(logits, dim=1).item()
            
            return prediction
        except Exception as e:
            logger.error(f"Error during text analysis: {str(e)}")
            return None

    def generate_text(self, prompt):
        try:
            inputs = self.gpt_tokenizer(
                prompt,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=self.config['model']['max_length']
            ).to(self.device)
            
            generated = self.model.generate_text(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=self.config['generation']['max_length']
            )
            
            generated_text = self.gpt_tokenizer.decode(
                generated[0],
                skip_special_tokens=True
            )
            
            return generated_text
        except Exception as e:
            logger.error(f"Error during text generation: {str(e)}")
            return None

def main():
    st.title("Smart Text Analysis and Generation System")
    

    app = TextAnalysisApp()
    
    
    st.sidebar.title("Options")
    task = st.sidebar.selectbox(
        "Choose Task",
        ["Text Analysis", "Text Generation", "Model Training"]
    )
    
    if task == "Text Analysis":
        st.header("Text Analysis")
        input_text = st.text_area("Enter text to analyze:")
        if st.button("Analyze"):
            if input_text:
                prediction = app.analyze_text(input_text)
                st.write(f"Prediction: {prediction}")
            else:
                st.warning("Please enter some text to analyze.")
    
    elif task == "Text Generation":
        st.header("Text Generation")
        prompt = st.text_area("Enter prompt for text generation:")
        if st.button("Generate"):
            if prompt:
                generated_text = app.generate_text(prompt)
                st.write("Generated Text:")
                st.write(generated_text)
            else:
                st.warning("Please enter a prompt for text generation.")
    
    elif task == "Model Training":
        st.header("Model Training")
        
    
        train_file = st.file_uploader("Upload training data (CSV)", type="csv")
        
        if train_file is not None:
            data = pd.read_csv(train_file)
            
            
            texts = data['text'].tolist()
            labels = data['label'].tolist()
            
            
            train_texts, val_texts, train_labels, val_labels = train_test_split(
                texts, labels, test_size=0.2, random_state=42
            )
            
            if st.button("Start Training"):
                with st.spinner("Training in progress..."):
                    fig = app.train_model(train_texts, train_labels, val_texts, val_labels)
                    if fig is not None:
                        st.success("Training completed successfully!")
                        st.pyplot(fig)
                    else:
                        st.error("Training failed. Check logs for details.")

if __name__ == "__main__":
    main()
