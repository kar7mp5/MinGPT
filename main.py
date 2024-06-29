# main.py
import torch
import torch.nn as nn
import torch.optim as optim
import tiktoken

from GPT import create_dataloader_v1, GPTModel


def generate_text(model, idx, max_new_tokens, context_size, tokenizer):
    """
    Generate text using a trained GPT model.

    Args:
        model (nn.Module): The GPT model used for text generation.
        idx (torch.Tensor): The input tensor containing token indices.
        max_new_tokens (int): The maximum number of new tokens to generate.
        context_size (int): The context size for the model's attention mechanism.

    Returns:
        torch.Tensor: The generated sequence of token indices.
    """
    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -context_size:]
            logits = model(idx_cond)
            logits = logits[:, -1, :]
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)   
            # Check for the end-of-sequence token and break if found
            if idx_next.squeeze().item() == tokenizer.encode('<eos>')[0]:
                break
            idx = torch.cat((idx, idx_next), dim=1)
    return idx


def get_text(model, device, PROMPT):
    tokenizer = tiktoken.get_encoding("gpt2")
    encoded = tokenizer.encode(PROMPT)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0).to(device)

    print(f"\n{'='*50}\n{' '*22}IN\n{'='*50}")
    print("\nInput text:", PROMPT)
    print("Encoded input text:", encoded)
    print("encoded_tensor.shape:", encoded_tensor.shape)

    out = generate_text(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=400,
        context_size=GPT_CONFIG_124M["context_length"],
        tokenizer=tokenizer
    )
    decoded_text = tokenizer.decode(out.squeeze(0).tolist())

    # Print the text only up to the end-of-sequence token
    eos_pos = decoded_text.find('<eos>')
    if eos_pos != -1:
        decoded_text = decoded_text[:eos_pos]

    # print(f"\n\n{'='*50}\n{' '*22}OUT\n{'='*50}")
    # print("\nOutput:", out)
    # print("Output length:", len(out[0]))
    # print("Output text:", decoded_text)

    return decoded_text


def generate_prompt(example):
    """
    Generate prompts and responses from the given example dataset.

    Args:
        example (dict): The dataset containing 'instruction' and 'output' keys.

    Returns:
        list: List of generated prompt strings.
    """
    output_texts = []
    for i in range(len(example['instruction'])):
        prompt = f"### Instruction: {example['instruction'][i]}\n\n### Response: {example['output'][i]}<eos>"
        output_texts.append(prompt)
    return output_texts


def train_model(model, dataloader, optimizer, criterion, num_epochs, device):
    """
    Train a GPT model.

    Args:
        model (nn.Module): The GPT model to be trained.
        dataloader (DataLoader): The DataLoader providing the training data.
        optimizer (torch.optim.Optimizer): The optimizer used for training.
        criterion (nn.Module): The loss function used for training.
        num_epochs (int): The number of epochs to train for.
        device (torch.device): The device (CPU or GPU) to perform training on.

    Returns:
        None
    """
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            input_ids, target_ids = batch
            input_ids, target_ids = input_ids.to(device), target_ids.to(device)
            outputs = model(input_ids)
            loss = criterion(outputs.view(-1, outputs.size(-1)), target_ids.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch {epoch + 1}, Loss: {avg_loss}')


def train(model, device, train_data):
    prompts = generate_prompt(train_data)
    print(prompts)
    dataloader = create_dataloader_v1(prompts, batch_size=10, max_length=512, stride=128, shuffle=False, drop_last=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=5e-5)
    num_epochs = 150
    train_model(model, dataloader, optimizer, criterion, num_epochs, device)



def process_output(output_text):
    processed_output = output_text.split('<eos>')[0].strip()
    return processed_output




if __name__ == "__main__":
    GPT_CONFIG_124M = {
        "vocab_size": 50257,
        "context_length": 1024,
        "emb_dim": 768,
        "n_heads": 12,
        "n_layers": 12,
        "drop_rate": 0.1,
        "qkv_bias": False
    }

    torch.manual_seed(0)
    model = GPTModel(GPT_CONFIG_124M)
    print(f"GPU? {torch.cuda.is_available()}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    example_data = {
        'instruction': [
            "Can you explain the BFS algorithm?",
        ],
        'output': [
            # BFS Algorithm in Python
            """
from collections import deque

def bfs(graph, start):
    visited = set()
    queue = deque([start])
    result = []

    while queue:
        vertex = queue.popleft()
        if vertex not in visited:
            visited.add(vertex)
            result.append(vertex)
            queue.extend(neighbor for neighbor in graph[vertex] if neighbor not in visited)
    return result

# Example usage:
graph = {
    'A': ['B', 'C'],
    'B': ['A', 'D', 'E'],
    'C': ['A', 'F'],
    'D': ['B'],
    'E': ['B', 'F'],
    'F': ['C', 'E']
}
print(bfs(graph, 'A'))  # Output: ['A', 'B', 'C', 'D', 'E', 'F']
            """
        ]
    }


    train(model, device, example_data)


    while True:
        user_input = input("입력을 하고 싶으신 내용을 입력해주세요 (종료하려면 'x' 입력): ")
        
        if user_input.lower() == 'x':
            break
        
        PROMPT = user_input
        formatted_prompt = f"### Instruction: {PROMPT}\n\n### Response:"

        output_text = get_text(model, device, formatted_prompt)
        # processed_output_text = process_output(output_text)
        print(output_text)
