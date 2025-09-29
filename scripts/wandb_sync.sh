# apt-get update && apt-get install -y inotify-tools
export WANDB_DIR=wandb
wandb sync --sync-all $WANDB_DIR
# inotifywait -m -r -e close_write,create,move $WANDB_DIR |
# while read path action file; do
#     wandb sync --sync-all $WANDB_DIR
# done