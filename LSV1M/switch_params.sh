if [ -d "./param_small" ]; then
    mv param param_full
    mv param_small param
else
    mv param param_small
    mv param_full param
fi
