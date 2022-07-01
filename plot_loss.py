import wandb
import seaborn as sns
import matplotlib.pyplot as plt

api = wandb.Api()

run = api.run("chcaa/gjallarhorn/254xlvgj")

loss = run.history(keys=["train_inner/loss", "valid/loss"])
loss = loss.set_index(loss["_step"])
loss = loss.drop("_step", axis=1)


p = sns.lineplot(data=loss)
p.set_xlabel("Steps")
p.set_ylabel("Loss")
plt.legend(labels=["Train", "Validation"])
plt.savefig("pretraining_loss", dpi=300)


sns.lineplot(data=loss, x="_step", y="train_inner/loss", color="blue")
sns.lineplot(data=loss, x="_step", y="valid/loss", color="orange")