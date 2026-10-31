import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import vm from "node:vm";
import test from "node:test";

// Minimal DOM stand-in: enough for BookText/TextFormat to build segment spans.
class FakeNode {
    constructor(tagName) {
        this.tagName = tagName;
        this.className = "";
        this.id = "";
        this.dataset = {};
        this.children = [];
        this._textContent = "";
    }
    appendChild(child) {
        this.children.push(child);
        return child;
    }
    set textContent(value) {
        this._textContent = value;
        this.children = [];
    }
    get textContent() {
        return this._textContent;
    }
}
const fakeDocument = {
    createElement: (tagName) => new FakeNode(tagName),
    createTextNode: (text) => ({ text }),
};

/** Serializes the fake tree; textSegment classes are omitted for brevity. */
function render(node) {
    if (node.text !== undefined) {
        return node.text;
    }
    const inner = node.children.length > 0
        ? node.children.map(render).join("")
        : node.textContent;
    const className = node.className.replace(/textSegment (hasAudio|noAudio)/, "seg").trim();
    return className ? `[${className}:${inner}]` : inner;
}

const repoRoot = fileURLToPath(new URL("../../", import.meta.url));
const context = vm.createContext({ console, document: fakeDocument });
for (const [file, name] of [["util.js", "Util"], ["text-format.js", "TextFormat"], ["book-text.js", "BookText"]]) {
    const source = readFileSync(`${repoRoot}/browser_player/${file}`, "utf8");
    vm.runInContext(`${source}\nglobalThis.${name} = ${name};`, context);
}
const { TextFormat, BookText } = context;
const plain = (value) => JSON.parse(JSON.stringify(value));

function makeBook(textSegments) {
    const bookText = Object.create(BookText.prototype);
    bookText.textSegments = textSegments;
    bookText.spans = [];
    bookText.segmentGroupSpans = [];
    return bookText;
}
const leaf = (text, fmt = null) => ({ text, fmt, time_start: 0, time_end: 1, playable: true });

test("parses markup into text and per-character masks", () => {
    assert.deepEqual(
        plain(TextFormat.parse("a <b>b<i>c</i></b> &amp;")),
        { text: "a bc &", masks: [0, 0, 1, 3, 0, 0] },
    );
});

test("rejects malformed markup and markup that does not match the leaf text", () => {
    // Stale or malformed formatting is ignored: the leaf renders as plain text.
    for (const fmt of ["<x>a</x>", "<b>a", "a</b>", "<b><i>a</b></i>", "a < b", "a & b"]) {
        assert.equal(TextFormat.parse(fmt), null, fmt);
    }
    assert.equal(TextFormat.masksFor("It was <i>cold</i>.", "It was warm."), null);
    assert.equal(TextFormat.masksFor("Unstyled.", "Unstyled."), null);
    assert.equal(TextFormat.masksFor(null, "Text."), null);
});

test("renders styled runs inside the segment span", () => {
    const bookText = makeBook([leaf("\nIt was very cold. ", "\nIt was <i>very <b>cold</b></i>. ")]);
    const block = new FakeNode("pre");

    bookText._appendFlatSegment(block, 0, false);

    assert.equal(render(block), "\n[seg:It was [fmtItalic:very ][fmtBold fmtItalic:cold].] ");
});

test("falls back to plain text for invalid formatting", () => {
    const bookText = makeBook([leaf("It was warm.", "It was <i>cold</i>.")]);
    const block = new FakeNode("pre");

    bookText._appendFlatSegment(block, 0, false);

    assert.equal(render(block), "[seg:It was warm.]");
});

test("whitespace between grouped leaves continues shared styles", () => {
    // An underline running across a phrase boundary is not broken by the
    // separator, which sits outside both leaf spans.
    const bookText = makeBook([
        leaf("She was very, ", "She was <u>very,</u> "),
        leaf("very cold.", "<u>very</u> cold."),
    ]);
    const block = new FakeNode("pre");

    bookText._appendSegmentGroup(block, { groupIndex: 0, startIndex: 0, endIndex: 2, leafCount: 2 }, false);

    assert.equal(
        render(block),
        "[segmentGroup hasAudio:[seg:She was [fmtUnderline:very,]][fmtUnderline: ][seg:[fmtUnderline:very] cold.]]",
    );
});
