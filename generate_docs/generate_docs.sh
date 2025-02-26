#!/bin/bash
# This file can be used to automatically generate the ../docs/ directory, which
# is used for the GutHub Pages website. See .README.md for instructions on how
# to use it.

cd ..
original_dirname=${PWD##*/}

# Rename directory to __moog_tmp__. This is necessary in case $original_dirname
# was 'moog', in which case pdoc won't work because there's a sub-directory
# called 'moog'
cd ..
rm -r __moog_tmp
mkdir __moog_tmp
cp -r $original_dirname/* __moog_tmp__/.
cd __moog_tmp__

# Create __init__.py file
rm __init__.py
printf '"""\n.. include:: README.md\n"""' >> __init__.py

# Replace relative links in README.md files
find . -name 'README.md' | while read filename; do
    echo "Doing link manipulation on '$filename'"

    # links
    dirname="https://github.com/jazlab/moog.github.io/blob/master/${filename%/*}/"
    sed_dirname=$(sed 's/\//\\\//g' <<< $dirname)
    sed -i '' "/](https:/ ! s/](/]($sed_dirname/g" $filename

    # Images
    sed -i '' "s/<img src=\"/<img src=\"$sed_dirname/g" $filename
    sed -i '' "s/\.gif\"/\.gif?raw=true\"/g" $filename
done

# Remove directories that mess up pdoc
rm -r docs
rm -r build
rm setup.py
rm -rf .git*

# Run the pdoc3 command
pdoc3 --html . --skip-errors --output-dir docs

# Clean up the docs, replacing strings to make them look nicer
cd docs
mv __moog_tmp__/* .
rm -r __moog_tmp__
find . -type f -exec sed -i '' 's/__moog_tmp__/modular_object_oriented_games/g' {} +
find . -type f -exec sed -i '' 's/modular_object_oriented_games.generate_docs/generate_docs/g' {} +
find . -type f -exec sed -i '' 's/modular_object_oriented_games.moog/moog/g' {} +
find . -type f -exec sed -i '' 's/modular_object_oriented_games.multi_agent_example/multi_agent_example/g' {} +
find . -type f -exec sed -i '' 's/modular_object_oriented_games.mworks/mworks/g' {} +
find . -type f -exec sed -i '' 's/modular_object_oriented_games.tests/tests/g' {} +
find . -type f -exec sed -i '' 's/modular_object_oriented_games/Homepage/g' {} +
cd ..

# Remove __init__.py file
rm __init__.py

# Copy docs back to $original_dirname
cd ..
rm -r $original_dirname/docs
cp -r __moog_tmp__/docs $original_dirname/
rm -r __moog_tmp__

# Launch website on localhost to preview
cd $original_dirname/docs
python -m http.server