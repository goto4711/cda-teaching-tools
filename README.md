# Cultural Data Analysis: teaching tools

Code-along exercises and interactive teaching tools for the Cultural Data Analysis lectures
(University of Amsterdam). Everything is static HTML, CSS and JavaScript: no build step.

Live site: https://goto4711.github.io/cda-teaching-tools/

```
index.html                  home page: weeks and tools
assets/course.css           course bar, help panel, presentation mode
assets/course.js            course bar, keyboard shortcuts, ?steps= and ?present= links
assets/guide.css            style for the guide pages

code-along/week0/ week1/ week3/ week4/   code-along exercises
workshops/week1/ week3/ week4/   student instructions for the workshops (Week 3: the full data critique)
notebooks/week1/ week3/ week4/   student workshop notebooks, opened in Colab from the workshop pages
code-along/week1/spectrum.html     Computer Programming Spectrum with agentic overlay
code-along/week0/stable-diffusion-demo.ipynb   Stable Diffusion demo for Colab (T4 GPU), opened from step 4

tools/cpu/          CPU simulator (Week 1)                 + guide.html
tools/loops/        Loops in four languages (Week 1)
tools/text/         Term frequency (Weeks 3 and 4)         + guide.html
tools/images/       How a computer sees an image (Week 3)
tools/neural-net/   Neural network training (Week 3)       + guide.html
tools/embeddings/   From words to embeddings (Week 4): landing page and multimodal.html
```

The Streamlit app "Term frequency vs. embeddings" stays in the `nlp-to-embedding` repository,
because Streamlit Community Cloud deploys from there. The AI ethics map stays in `ai-ethics-histories`.

## Workshop notebooks

The workshop pages (`workshops/`) link to the student versions in `notebooks/` through Colab
(`https://colab.research.google.com/github/goto4711/cda-teaching-tools/blob/main/notebooks/...`).
Every notebook starts with a box telling students to use **File → Save a copy in Drive** first,
because Colab does not keep changes to a notebook opened from GitHub. Solutions are never put here.
The notebooks download their data and checks from the `cdai` repository.

## Shortcuts and links (all tools)

| Key | Action |
|---|---|
| Space | Step (what a step is depends on the tool) |
| Enter | Run / pause, where the tool has one |
| R | Reset |
| P | Presentation mode: larger, shows the page address; + and − change the size |
| G | Guide, where there is one |
| ? | Help |

URL parameters: `?present=1` everywhere; `?steps=N` wherever there is a Step button.
Tool-specific:
- `tools/cpu/?program=add|loop|input`
- `tools/loops/?lang=assembly|python|javascript|cpp` (that language first, outlined)
- `tools/text/?example=fox|bites|house`, `?d1=…&d2=…&d3=…&stop=…`, `&mode=tfidf`
- `tools/images/?grid=8|16|32`, `?source=random`, `?seed=42`
- `tools/neural-net/?seed=1`, `?data=clear`
- `tools/embeddings/multimodal.html?trained=1`, `?select=Feline` (a word) or `?select=cat` (a picture)

## Adding a tool

1. Put it in `tools/<name>/`.
2. Before `</head>`: `<link rel="stylesheet" href="../../assets/course.css">`.
3. Before `</body>`:
   ```html
   <script src="../../assets/course.js" data-week="3" data-title="My tool"
       data-step="#step-button" data-reset="#reset-button" data-guide="guide.html"></script>
   ```
   See the comment at the top of `assets/course.js` for all options.
4. Add a card to the Interactive tools section of `index.html`.

The neural network page is pre-compiled: edit `tools/neural-net/nn_app.jsx`, then compile it with Babel
(preset-react) and paste the result into the `<script>` in `index.html`.

## Publishing with GitHub Pages

Settings → Pages → Deploy from a branch → `main`, `/ (root)`.

## Installing the course libraries locally

The requirements file lives in the goto4711/cdai repository, next to the course data.

```
pip install -r https://raw.githubusercontent.com/goto4711/cdai/refs/heads/main/requirements.txt
pip install --no-build-isolation git+https://github.com/Kaggle/learntools.git
```

`learntools` is installed separately because it can only be built once pandas is installed.
Tesseract (Week 5) and Graphviz (Week 3) are separate programs. The notebooks were written for
pandas 2 and transformers 4, so both are capped below their next major version.
