import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import vm from "node:vm";
import test from "node:test";

const context = vm.createContext({ console });
const source = readFileSync(new URL("../browser_player/navigation-panel.js", import.meta.url), "utf8");
vm.runInContext(`${source}\nglobalThis.NavigationPanel = NavigationPanel;`, context);

function makeItem() {
    const classes = new Set();
    const attributes = new Map();
    return {
        classList: { add: name => classes.add(name), remove: name => classes.delete(name) },
        setAttribute: (name, value) => attributes.set(name, value),
        removeAttribute: name => attributes.delete(name),
        classes,
        attributes,
    };
}

function makePanel() {
    const panel = Object.create(context.NavigationPanel.prototype);
    panel.sections = [
        { start_index: 0, end_index: 3 },
        { start_index: 3, end_index: 6 },
    ];
    panel.sectionItemEls = [makeItem(), makeItem()];
    panel.activeSectionIndex = -1;
    return panel;
}

test("section indicator follows enclosing half-open ranges and clears outside them", () => {
    const panel = makePanel();
    const [first, second] = panel.sectionItemEls;
    for (const index of [0, 2]) {
        panel.updateCurrentSection(index);
        assert.equal(first.classes.has("currentSection"), true);
        assert.equal(first.attributes.get("aria-current"), "location");
        assert.equal(second.classes.has("currentSection"), false);
    }
    panel.updateCurrentSection(3);
    assert.equal(first.classes.has("currentSection"), false);
    assert.equal(first.attributes.has("aria-current"), false);
    assert.equal(second.classes.has("currentSection"), true);
    panel.updateCurrentSection(6);
    assert.equal(second.classes.has("currentSection"), false);
    panel.updateCurrentSection(1);
    panel.updateCurrentSection(-1);
    assert.equal(first.classes.has("currentSection"), false);
});

test("rebuilding sections resets the previous file's indicator", () => {
    const panel = makePanel();
    panel.updateCurrentSection(4);
    panel.sectionList = { replaceChildren() {}, appendChild() {} };
    panel._makeSectionItemEl = makeItem;
    panel._updateSectionList();
    assert.equal(panel.activeSectionIndex, -1);
    panel.updateCurrentSection(1);
    assert.equal(panel.sectionItemEls[0].classes.has("currentSection"), true);
    assert.equal(panel.sectionItemEls[1].classes.has("currentSection"), false);
});

test("documents without sections have no indicator", () => {
    const panel = makePanel();
    panel.sections = [];
    panel.sectionItemEls = [];
    panel.updateCurrentSection(0);
    assert.equal(panel.activeSectionIndex, -1);
});
