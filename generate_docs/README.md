# Automatically Generating Documentation

The files in [`./docs`](../../docs) are automatically generated and are used for
the [GitHub Pages website](https://jazlab.github.io/moog.github.io/). They are
generated using [pdoc3](https://pdoc3.github.io/generate_docs/), and should be
re-generated upon any changes to the codebase.

To re-generate the documentation, follow these steps:
1. If you do have `pdoc3` installed, then install it with `$ pip install pdoc3`.
2. Navigate to this directory and run `$ bash generate_docs.sh`]. If any errors
   arise, please do not git commit --- instead, please do due diligence
   understanding the [`generate_docs.sh`](../generate_docs.sh) and making sure
   that once you've gotten it working you haven't introduced any unwanted
   changes.
3. Visit the localhost URL printed by step (2) and check that it looks okay and
   some of the sidebar links work. Do not be alarmed if the images do not appear
   or the links in the main text do not work --- that is expected.
4. Go into [`../docs/index.html`](../../docs/index.html) and remove a line near
   the top that looks like `<h1 class="title">Package
   <code>Homepage</code></h1>`, as well as one `<header>` line above or below
   it. This should be lines 21 and 22. This is just a cosmetic change to avoid a
   double-header on the website. Save the file.
5. Refresh localhost or re-generate the localshot website by navigating to
   `../docs` and running `$ python -m http.server` to again preview the website
   on a localhost.
6. If everything looks okay, commit.
