// -----------------------------
// STATIC DEMO CONFIG
// -----------------------------

const DEMO_FILES = {
  "wireless headphones for studying": "demo-data/wireless_headphones.json",
  "gaming monitor": "demo-data/gaming_monitor.json",
  "iron": "demo-data/iron.json",
};

// -----------------------------
// DOM ELEMENTS
// -----------------------------

const form = document.querySelector("#search-form");
const input = document.querySelector("#search-input");
const chipButtons = document.querySelectorAll(".query-chip");
const statusRegion = document.querySelector("#status-region");
const overviewSection = document.querySelector("#overview-section");
const overviewText = document.querySelector("#overview-text");
const whyBestText = document.querySelector("#why-best-text");
const quickPicks = document.querySelector("#quick-picks");
const resultsGrid = document.querySelector("#results-grid");
const resultCount = document.querySelector("#result-count");
const cardTemplate = document.querySelector("#product-card-template");

// -----------------------------
// EVENT LISTENERS
// -----------------------------

form.addEventListener("submit", (event) => {
  event.preventDefault();
  runSearch(input.value);
});

chipButtons.forEach((chip) => {
  chip.addEventListener("click", () => {
    const query = chip.dataset.query || chip.textContent.trim();
    input.value = query;
    runSearch(query);
  });
});

window.addEventListener("DOMContentLoaded", () => {
  resultsGrid.append(
    createEmptyState("Loading saved demo recommendations...")
  );

  const defaultQuery = "wireless headphones for studying";
  input.value = defaultQuery;
  runSearch(defaultQuery);
});

// -----------------------------
// SEARCH LOGIC
// -----------------------------

function normalizeQuery(query) {
  return String(query || "").trim().toLowerCase();
}

function resolveDemoFile(query) {
  const normalized = normalizeQuery(query);

  const aliases = {
    "wireless headphones": "wireless headphones for studying",
    "wireless headphones for studying": "wireless headphones for studying",

    "gaming monitor": "gaming monitor",
    "gaming monitors": "gaming monitor",
    "curved monitor for gaming": "gaming monitor",

    "steam iron": "steam iron for clothes",
    "clothes iron": "steam iron for clothes",
    "iron for clothes": "steam iron for clothes",
    "steam iron for clothes": "steam iron for clothes",

    // "ergonomic mouse": "wireless headphones for studying",
    // "ergonomic mouse for wrist pain": "wireless headphones for studying",
  };

  const resolvedQuery = aliases[normalized] || normalized;
  return DEMO_FILES[resolvedQuery];
}

async function getSearchResults(query) {
  const demoFile = resolveDemoFile(query);

  if (!demoFile) {
    throw new Error(
      "No saved demo data is available for this query. Please select one of the demo buttons."
    );
  }

  const response = await fetch(demoFile);

  if (!response.ok) {
    throw new Error(`Could not load saved demo data: ${demoFile}`);
  }

  return await response.json();
}

async function runSearch(rawQuery) {
  const query = String(rawQuery || "").trim();

  if (!query) {
    setStatus("Enter a search query or select one of the saved demo examples.", "error");
    input.focus();
    return;
  }

  setLoading(true);
  clearResults();
  setStatus(`Loading saved recommendations for "${query}"...`, "loading");

  try {
    const data = await getSearchResults(query);
    renderResponse(data);
    setStatus("");
  } catch (error) {
    overviewSection.classList.add("is-hidden");
    resultsGrid.innerHTML = "";
    resultsGrid.append(createEmptyState("No demo results could be loaded."));
    setStatus(error.message, "error");
  } finally {
    setLoading(false);
  }
}

// -----------------------------
// RENDERING
// -----------------------------

function renderResponse(data) {
  const results = Array.isArray(data.results) ? data.results : [];

  overviewText.textContent = data.overview || "No overview was returned for this query.";
  whyBestText.textContent =
    data.why_best_overall || "No best-result explanation was returned.";

  resultCount.textContent = `${results.length} result${results.length === 1 ? "" : "s"}`;

  renderQuickPicks(data);

  overviewSection.classList.remove("is-hidden");

  resultsGrid.innerHTML = "";

  if (!results.length) {
    resultsGrid.append(createEmptyState("No products were returned for this query."));
    return;
  }

  results.forEach((product) => {
    resultsGrid.append(renderProductCard(product));
  });
}

function renderQuickPicks(data) {
  quickPicks.innerHTML = "";

  const picks = [
    { label: "Best overall", item: data.best_overall },
    { label: "Best value", item: data.best_value },
    ...normalizeSpecializedPicks(data.specialized_picks),
  ].filter((pick) => pick.item);

  if (!picks.length) {
    return;
  }

  picks.forEach((pick) => {
    quickPicks.append(renderPickCard(pick.label, pick.item, pick.note));
  });
}

function normalizeSpecializedPicks(specializedPicks) {
  if (!specializedPicks) {
    return [];
  }

  if (Array.isArray(specializedPicks)) {
    return specializedPicks.map((entry, index) =>
      normalizePickEntry(entry, `Specialized ${index + 1}`)
    );
  }

  if (typeof specializedPicks === "object") {
    return Object.entries(specializedPicks).map(([label, entry]) =>
      normalizePickEntry(entry, humanizeLabel(label))
    );
  }

  return [];
}

function normalizePickEntry(entry, fallbackLabel) {
  if (!entry || typeof entry !== "object") {
    return {
      label: fallbackLabel,
      item: entry,
    };
  }

  return {
    label: entry.pick_label || entry.label || entry.category || fallbackLabel,
    item: entry.product || entry.item || entry.pick || entry,
    note: entry.reason || entry.why || entry.explanation || entry.rationale,
  };
}

function renderPickCard(label, item, note = "") {
  const card = createElement("article", "pick-card");
  const labelElement = createElement("span", "pick-label", label);
  const title = createElement("p", "pick-title", getProductTitle(item));

  const meta = createElement(
    "p",
    "pick-meta",
    `${formatPrice(item)} • ${formatMatch(item.final_score)} match`
  );

  card.append(labelElement, title, meta);

  if (note) {
    card.append(createElement("p", "pick-meta", note));
  }

  return card;
}

function renderProductCard(product) {
  const fragment = cardTemplate.content.cloneNode(true);

  const card = fragment.querySelector(".product-card");
  const imageWrap = fragment.querySelector(".image-wrap");
  const title = fragment.querySelector(".product-title");
  const matchBadge = fragment.querySelector(".match-badge");
  const meta = fragment.querySelector(".product-meta");
  const featureList = fragment.querySelector(".feature-list");
  const detailsList = fragment.querySelector(".match-details dl");

  imageWrap.append(renderProductImage(product));
  title.textContent = getProductTitle(product);
  matchBadge.textContent = formatMatch(product.final_score);

  meta.append(...renderMeta(product));

  renderFeatureChips(product.key_features).forEach((chip) => {
    featureList.append(chip);
  });

  renderDetails(product).forEach(([label, value]) => {
    detailsList.append(
      createElement("dt", "", label),
      createElement("dd", "", value)
    );
  });

  if (product.parent_asin) {
    card.dataset.asin = product.parent_asin;
  }

  return fragment;
}

function renderProductImage(product) {
  const title = getProductTitle(product);

  if (!product.image_url) {
    return renderImageFallback(title);
  }

  const image = document.createElement("img");
  image.src = product.image_url;
  image.alt = title;
  image.loading = "lazy";
  image.referrerPolicy = "no-referrer";

  image.addEventListener("error", () => {
    image.replaceWith(renderImageFallback(title));
  });

  return image;
}

function renderImageFallback(title) {
  const fallback = createElement("div", "image-fallback", getInitials(title));
  fallback.setAttribute("aria-label", "Product image unavailable");
  return fallback;
}

function renderMeta(product) {
  const nodes = [];

  nodes.push(createElement("span", "meta-strong", formatPrice(product)));

  if (isPresent(product.avg_rating)) {
    nodes.push(
      createElement("span", "rating", `${formatNumber(product.avg_rating, 1)} ★`)
    );
  }

  if (isPresent(product.review_count)) {
    nodes.push(
      createElement("span", "", `${formatInteger(product.review_count)} reviews`)
    );
  }

  return nodes;
}

function renderFeatureChips(features) {
  const normalized = Array.isArray(features)
    ? features.filter(Boolean)
    : String(features || "")
        .split(",")
        .map((feature) => feature.trim())
        .filter(Boolean);

  const visibleFeatures = normalized.length
    ? normalized.slice(0, 5)
    : ["explainable rank"];

  return visibleFeatures.map((feature) =>
    createElement("span", "feature-chip", String(feature))
  );
}

function renderDetails(product) {
  return [
    ["Semantic score", formatScore(product.semantic_score)],
    ["ALS score", formatScore(product.als_score)],
    ["Hybrid score", formatScore(product.hybrid_score)],
    ["Final score", formatScore(product.final_score)],
    ["Value score", formatPercent(product.affordability_score)],
    ["Value label", product.affordability_label || "Not provided"],
    ["Verified purchases", formatPercent(product.verified_purchase_ratio)],
    ["Trust label", product.trust_label || "Not provided"],
  ];
}

// -----------------------------
// UI HELPERS
// -----------------------------

function clearResults() {
  overviewSection.classList.add("is-hidden");
  quickPicks.innerHTML = "";
  resultsGrid.innerHTML = "";
  resultCount.textContent = "";
}

function createEmptyState(message) {
  return createElement("div", "empty-state", message);
}

function setStatus(message, type = "info") {
  statusRegion.innerHTML = "";

  if (!message) {
    return;
  }

  const card = document.createElement("div");
  card.className = `status-card ${type}`;

  if (type === "loading") {
    const row = document.createElement("div");
    row.className = "loading-row";
    row.append(
      createElement("span", "spinner"),
      document.createTextNode(message)
    );
    card.append(row);
  } else {
    card.textContent = message;
  }

  statusRegion.append(card);
}

function setLoading(isLoading) {
  const button = form.querySelector("button");

  button.disabled = isLoading;
  input.disabled = isLoading;

  chipButtons.forEach((chip) => {
    chip.disabled = isLoading;
  });

  button.textContent = isLoading ? "Loading..." : "Search";
}

// -----------------------------
// FORMATTERS
// -----------------------------

function getProductTitle(product) {
  return product?.display_title || product?.title || "Untitled product";
}

function formatPrice(product) {
  if (isPresent(product?.price)) {
    if (typeof product.price === "number") {
      return new Intl.NumberFormat("en-US", {
        style: "currency",
        currency: "USD",
        maximumFractionDigits: 2,
      }).format(product.price);
    }

    return String(product.price);
  }

  return product?.price_bucket || "Price not listed";
}

function formatMatch(value) {
  if (!isPresent(value)) {
    return "N/A";
  }

  const numeric = Number(value);
  const percent = numeric <= 1 ? numeric * 100 : numeric;

  return `${Math.round(percent)}%`;
}

function formatScore(value) {
  if (!isPresent(value)) {
    return "N/A";
  }

  return formatNumber(Number(value), 3);
}

function formatPercent(value) {
  if (!isPresent(value)) {
    return "N/A";
  }

  const numeric = Number(value);
  const percent = numeric <= 1 ? numeric * 100 : numeric;

  return `${formatNumber(percent, 1)}%`;
}

function formatNumber(value, digits = 2) {
  if (!Number.isFinite(Number(value))) {
    return "N/A";
  }

  return new Intl.NumberFormat("en-US", {
    maximumFractionDigits: digits,
    minimumFractionDigits: digits,
  }).format(Number(value));
}

function formatInteger(value) {
  if (!Number.isFinite(Number(value))) {
    return "N/A";
  }

  return new Intl.NumberFormat("en-US").format(Number(value));
}

function getInitials(title) {
  const words = String(title)
    .replace(/[^a-zA-Z0-9 ]/g, " ")
    .split(" ")
    .filter(Boolean)
    .slice(0, 2);

  return words.map((word) => word[0].toUpperCase()).join("") || "PS";
}

function humanizeLabel(label) {
  return String(label)
    .replace(/[_-]+/g, " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function createElement(tagName, className = "", text = "") {
  const element = document.createElement(tagName);

  if (className) {
    element.className = className;
  }

  if (text) {
    element.textContent = text;
  }

  return element;
}

function isPresent(value) {
  return value !== undefined && value !== null && value !== "";
}