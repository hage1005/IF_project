search_dir=$1
for entry in "$search_dir"/*
do
  echo "$entry"
  python3 fenchel_CIFAR_main.py --YAMLPath "$entry" &
done