#!/data/data/com.termux/files/usr/bin/bash
# Resilient night runner: trains one epoch per invocation, survives phone
# freezes/kills. Checkpoint resumes from where it stopped.
cd /root/devops/rogue6/app || exit 1
for ep in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do
  if [ -f ../vault/brain/epoch_done_$ep ]; then
    echo "[night] epoch $ep already done, skipping"
    continue
  fi
  echo "[night] starting epoch $ep at $(date +%H:%M)"
  python3 -u trainer.py --single "$ep" >> ../vault/train.log 2>&1
  if grep -q "epoch $ep/20" ../vault/train.log; then
    touch ../vault/brain/epoch_done_$ep
    echo "[night] epoch $ep DONE"
  else
    echo "[night] epoch $ep died, will retry next loop"
  fi
  sleep 5
done
echo "[night] ALL EPOCHS COMPLETE"
