
#!/usr/bin/env bash
#
# rename_aligned.sh — renomeia CJ_LiF_*_re_aligned_aligned.dat → CJ_LiF_*_aligned.dat
#

# verifica se há arquivos a renomear
shopt -s nullglob
files=(CJ_LiF_*_re_aligned_aligned.dat)
if [ ${#files[@]} -eq 0 ]; then
  echo "Nenhum arquivo *_re_aligned_aligned.dat encontrado neste diretório."
  exit 0
fi

# para cada arquivo, faz a troca do sufixo
for f in "${files[@]}"; do
  # destina sem o sufixo duplo
  dest="${f%_re_aligned_aligned.dat}_aligned.dat"
  echo "Renomeando '$f' → '$dest'"
  mv -- "$f" "$dest"
done

