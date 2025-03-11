from requirements import *
from preprocess import *

model = AttentionUnet(
    spatial_dims=3,  
    in_channels=1,  
    out_channels=3,  
    channels=(16, 32, 64, 128, 256),  
    strides=(2, 2, 2, 2), 
    dropout=0.3  
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

print(model)

# %%
def combined_loss(pred, target):
    ce_loss = nn.CrossEntropyLoss()(pred, target)
    return ce_loss 

optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=5e-4) 

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.3, patience=3)

# %%
# Function to save checkpoint
def save_checkpoint(model, optimizer, scaler, epoch, loss, filepath="checkpoint.pth"):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "loss": loss,
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved at epoch {epoch+1}")

# Function to load checkpoint
def load_checkpoint(model, optimizer, scaler, filepath="checkpoint.pth"):
    if os.path.exists(filepath):
        checkpoint = torch.load(filepath, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_loss = checkpoint["loss"]
        print(f"Resuming from epoch {start_epoch} with loss {best_loss:.4f}")
        return model, optimizer, scaler, start_epoch, best_loss
    else:
        print("No checkpoint found, starting fresh!")
        return model, optimizer, scaler, 0, float("inf") 

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

scaler = torch.amp.GradScaler()

best_val_loss = float("inf")
early_stopping_counter = 0
patience = 2  

num_epochs = 10 
grad_accumulation_steps = 4 
best_loss = float("inf") 

model, optimizer, scaler, start_epoch, best_loss = load_checkpoint(model, optimizer, scaler)

for epoch in range(start_epoch, num_epochs):
    model.train()
    epoch_loss = 0

    loop = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)

    for batch_idx, (images, masks) in enumerate(loop):
        images = images.to(device, dtype=torch.float32)
        masks = masks.to(device, dtype=torch.long) 
        masks = masks.squeeze(1) 

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            outputs = model(images)  # Forward pass
            loss = combined_loss(outputs, masks)

        loss = loss / grad_accumulation_steps
        scaler.scale(loss).backward()

        if (batch_idx + 1) % grad_accumulation_steps == 0 or batch_idx == len(train_dataloader) - 1:
            scaler.step(optimizer) 
            scaler.update()
            optimizer.zero_grad(set_to_none=True) 

        epoch_loss += loss.item()
        loop.set_postfix(loss=loss.item()) 

        # Free up memory
        del images, masks, outputs, loss
        torch.cuda.empty_cache()

    avg_train_loss = epoch_loss / len(train_dataloader)
    print(f"Epoch {epoch+1}/{num_epochs}, Avg Training Loss: {avg_train_loss:.4f}")


    if (epoch + 1) % 5 == 0 or avg_train_loss < best_loss:
        save_checkpoint(model, optimizer, scaler, epoch, avg_train_loss, "checkpoint.pth")
        best_loss = min(best_loss, avg_train_loss) 

    # **Validation Loop**
    model.eval()
    total_val_loss = 0.0

    with torch.no_grad():
        val_loop = tqdm(val_dataloader, desc=f"Validating...", leave=False)
        for images, masks in val_loop:
            images = images.to(device, dtype=torch.float32)
            masks = masks.to(device, dtype=torch.long)
            masks = masks.squeeze(1)

            outputs = model(images)
            val_loss = combined_loss(outputs, masks)
            total_val_loss += val_loss.item()

            del images, masks, outputs, val_loss
            torch.cuda.empty_cache()

    avg_val_loss = total_val_loss / len(val_dataloader)
    print(f"Epoch {epoch+1}: Validation Loss: {avg_val_loss:.4f}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        early_stopping_counter = 0
        torch.save(model.state_dict(), "best_model.pth")
        print(" Model improved. Saving best model.")
    else:
        early_stopping_counter += 1
        print(f" Validation loss did not improve ({early_stopping_counter}/{patience})")

    if early_stopping_counter >= patience:
        print(" Early stopping triggered. Stopping training.")
        break 

    scheduler.step(avg_val_loss)

print("Training complete!")