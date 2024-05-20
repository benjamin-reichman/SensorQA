from transformers import T5ForConditionalGeneration, T5Tokenizer
from sensorqa_dataset import SensorQADataset, generic_padded_collator
import torch
from tqdm import tqdm
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR


def model_loop(epoch, model, val_dataloader):
    model.eval()
    with tqdm(total=len(val_dataloader)) as pbar:
        for i, batch in enumerate(dataloaders[mode]):
            outputs = model.generate(batch["input_ids"].cuda())
            output_answer = dataloaders[mode].dataset.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            for k, j in enumerate(batch["indices"]):
                dataloaders[mode].dataset[j] = output_answer[k]
            pbar.update(1)
        val_dataloader.dataset.save(f"sensorqa_training_outputs/question_only_training_results_epoch_{epoch}.json")

tmodel = T5ForConditionalGeneration.from_pretrained("t5-base").cuda()

datasets = {}
dataloaders = {}
for mode in ['train', 'val']:
    datasets[mode] = SensorQADataset(f"../../2023_task_files/sensorqa/overall_sensorqa_dataset_{mode}.json", T5Tokenizer, ["t5-base"])
    dataloaders[mode] = torch.utils.data.DataLoader(datasets[mode], batch_size=32, shuffle=False, drop_last=False, num_workers=12, pin_memory=True, collate_fn=generic_padded_collator)

hyperparameters = {
    "learning_rate": 1e-5,
}

tmodel.train()
optimizer = AdamW(tmodel.parameters(), lr=hyperparameters["learning_rate"])

num_epochs = 100
for epoch in range(num_epochs+1):
    epoch_loss = 0
    if epoch % 10 == 0:
        model_loop(epoch, tmodel, dataloaders["val"])
    with tqdm(total=len(dataloaders["train"])) as pbar:
        for i, batch in enumerate(dataloaders["train"]):
            tmodel.zero_grad(set_to_none=True)
            outputs = tmodel(batch["input_ids"].cuda(), batch["attention_mask"].cuda(), labels=batch["labels"].cuda())
            loss = outputs.loss
            loss.backward()
            epoch_loss += loss.item()
            torch.nn.utils.clip_grad_norm_(tmodel.parameters(), 2.0)
            optimizer.step()
            if i == len(dataloaders["train"])-1:
                print("Epoch {}: Loss: {:.4f}".format(epoch, epoch_loss/len(dataloaders["train"])))
            pbar.update(1)

