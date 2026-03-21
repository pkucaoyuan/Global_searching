# Find Icon - Icon Library Search

Search and use icons from the `.claude/icons/` library for TikZ diagrams.

## Arguments

- `$ARGUMENTS` - Search query (e.g., "robot", "database", "arrow")

## Icon Library

The library contains **46 icons** in these categories:

### AI/ML (`ai_ml/`)
| Icon | File | Use for |
|------|------|---------|
| 🧠 | `brain.pdf` | Neural network, intelligence, thinking |
| 🤖 | `robot.pdf` | AI agent, bot, automation |
| 💻 | `chip.pdf` | Computing, hardware, processor |
| 🔗 | `network.pdf` | Graph, connections, topology |
| ✨ | `sparkles.pdf` | AI magic, generation, enhancement |

### Data (`data/`)
| Icon | File | Use for |
|------|------|---------|
| 🗄️ | `database.pdf` | Data storage, records |
| 📊 | `chart.pdf` | Bar chart, visualization |
| 📈 | `trending.pdf` | Growth, trends, increase |
| 🥧 | `pie_chart.pdf` | Proportions, distribution |
| 📋 | `table.pdf` | Structured data, matrix |

### Cloud (`cloud/`)
| Icon | File | Use for |
|------|------|---------|
| ☁️ | `cloud.pdf` | Cloud computing, storage |
| 🖥️ | `server.pdf` | Server, backend, infrastructure |
| 📡 | `network.pdf` | Connectivity, wireless |
| 🌍 | `globe.pdf` | Global, internet, worldwide |

### Business (`business/`)
| Icon | File | Use for |
|------|------|---------|
| 👤 | `user.pdf` | Person, agent, customer |
| 👥 | `users.pdf` | Group, population, crowd |
| 🏢 | `building.pdf` | Company, organization |
| 💰 | `money.pdf` | Cost, revenue, pricing |
| 💼 | `briefcase.pdf` | Business, work |
| 🏭 | `factory.pdf` | Manufacturing, production |
| 🚚 | `truck.pdf` | Logistics, delivery, transportation |
| 📦 | `package.pdf` | Product, order, item |
| 🏪 | `warehouse.pdf` | Inventory, storage |

### Science (`science/`)
| Icon | File | Use for |
|------|------|---------|
| ⚛️ | `atom.pdf` | Physics, fundamental, core |
| 🧪 | `flask.pdf` | Experiment, chemistry |
| 🔬 | `microscope.pdf` | Analysis, detailed study |
| 🧫 | `test_tube.pdf` | Testing, samples |

### Arrows (`arrows/`)
| Icon | File | Use for |
|------|------|---------|
| → | `arrow_right.pdf` | Flow, direction, next |
| ← | `arrow_left.pdf` | Back, previous |
| ↑ | `arrow_up.pdf` | Increase, upload |
| ↓ | `arrow_down.pdf` | Decrease, download |
| ↔ | `bidirectional.pdf` | Exchange, two-way |
| ⟳ | `loop.pdf` | Iteration, cycle, repeat |
| ⑂ | `branch.pdf` | Split, diverge, options |
| ⑃ | `merge.pdf` | Combine, aggregate |
| ➡ | `flow_right.pdf` | Process flow |

### Misc (`misc/`)
| Icon | File | Use for |
|------|------|---------|
| 💡 | `lightbulb.pdf` | Idea, insight, innovation |
| 🎯 | `target.pdf` | Goal, objective, accuracy |
| ⏰ | `clock.pdf` | Time, scheduling, delay |
| 📅 | `calendar.pdf` | Date, planning |
| 📄 | `document.pdf` | Paper, file, report |
| 🔍 | `search.pdf` | Search, query, find |
| ⚙️ | `settings.pdf` | Configuration, parameters |
| ✅ | `check.pdf` | Success, correct, valid |
| ❌ | `cross.pdf` | Error, wrong, invalid |
| ⚠️ | `warning.pdf` | Warning, caution, alert |

## Usage Examples

### Basic TikZ Node

```latex
\usepackage{tikz}
\usepackage{graphicx}

\begin{tikzpicture}
  \node at (0,0) {\includegraphics[width=1cm]{.claude/icons/pdf/ai_ml/brain.pdf}};
\end{tikzpicture}
```

### Flow Diagram

```latex
\begin{tikzpicture}[
    icon/.style={inner sep=0pt},
    arrow/.style={->, thick, >=stealth}
]
  % Icons
  \node[icon] (data) at (0,0) {\includegraphics[width=1cm]{.claude/icons/pdf/data/database.pdf}};
  \node[icon] (model) at (3,0) {\includegraphics[width=1cm]{.claude/icons/pdf/ai_ml/brain.pdf}};
  \node[icon] (output) at (6,0) {\includegraphics[width=1cm]{.claude/icons/pdf/data/chart.pdf}};

  % Arrows
  \draw[arrow] (data) -- (model);
  \draw[arrow] (model) -- (output);

  % Labels
  \node[below=2mm] at (data) {Data};
  \node[below=2mm] at (model) {Model};
  \node[below=2mm] at (output) {Results};
\end{tikzpicture}
```

### System Architecture

```latex
\begin{tikzpicture}
  % Cloud layer
  \node[icon] (cloud) at (3,4) {\includegraphics[width=1.5cm]{.claude/icons/pdf/cloud/cloud.pdf}};

  % Server layer
  \node[icon] (s1) at (1,2) {\includegraphics[width=0.8cm]{.claude/icons/pdf/cloud/server.pdf}};
  \node[icon] (s2) at (3,2) {\includegraphics[width=0.8cm]{.claude/icons/pdf/cloud/server.pdf}};
  \node[icon] (s3) at (5,2) {\includegraphics[width=0.8cm]{.claude/icons/pdf/cloud/server.pdf}};

  % User layer
  \node[icon] (u1) at (2,0) {\includegraphics[width=0.8cm]{.claude/icons/pdf/business/user.pdf}};
  \node[icon] (u2) at (4,0) {\includegraphics[width=0.8cm]{.claude/icons/pdf/business/user.pdf}};

  % Connections
  \draw[dashed] (cloud) -- (s1);
  \draw[dashed] (cloud) -- (s2);
  \draw[dashed] (cloud) -- (s3);
  \draw (s1) -- (u1);
  \draw (s2) -- (u1);
  \draw (s2) -- (u2);
  \draw (s3) -- (u2);
\end{tikzpicture}
```

## Scripts

The library includes helper scripts:

```bash
# List all icons
.claude/icons/index.sh

# Search icons
.claude/icons/index.sh brain

# Download new icon from Lucide
.claude/icons/download.sh rocket ai_ml

# Convert all SVGs to PDF
.claude/icons/convert.sh
```

## Auto-Download New Icons

If the requested icon doesn't exist, I will:

1. Search [Lucide Icons](https://lucide.dev/icons/) for matching icons
2. Download using: `.claude/icons/download.sh <icon_name> <category>`
3. Provide the TikZ code

**Available Lucide icons**: 1400+ icons at https://lucide.dev/icons/

## Workflow

1. **Search**: Describe what you need (e.g., "I need an icon for a neural network")
2. **Check Library**: Look in `.claude/icons/pdf/` for existing icons
3. **Download if Missing**: Use `download.sh` to fetch from Lucide
4. **Provide Code**: Give ready-to-use TikZ snippet

## Begin

Based on `$ARGUMENTS`:

1. First, run `Bash: .claude/icons/index.sh` to see available icons
2. If icon exists → provide TikZ code
3. If icon missing → download from Lucide using `download.sh`
4. Suggest alternatives if exact match unavailable
