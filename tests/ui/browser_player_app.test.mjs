import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import vm from "node:vm";
import test from "node:test";

const context = vm.createContext({ console });
const source = readFileSync(new URL("../../browser_player/app.js", import.meta.url), "utf8");
vm.runInContext(`${source}\nglobalThis.App = App;`, context);

function makeApp(highlightChanged = false) {
    const calls = [];
    const app = Object.create(context.App.prototype);
    app.zombieChecker = { isChecking: false };
    app.toast = { isShowingPlayPrompt: () => false };
    app.rootAttributer = { set: () => {} };
    app.audio = { currentTime: 12.5, ended: true };
    app.playerVisibility = { updateVisibility() {} };
    app.navigationPanel = {
        updateCurrentSection(index) {
            calls.push(["section", index]);
        },
    };
    app.bookText = {
        currentIndex: 3,
        getCurrentIndex() { return this.currentIndex; },
        updateHighlight(time) {
            calls.push(["highlight", time]);
            return highlightChanged;
        },
        scrollToIndex(index) {
            calls.push(["scroll", index]);
        },
    };
    return { app, calls };
}

test("resuming re-centers the unchanged highlighted segment", () => {
    const { app, calls } = makeApp();
    app.onAudioPlay();
    assert.deepEqual(calls, [["highlight", 12.5], ["scroll", 3], ["section", 3]]);
});

test("playing after a seek lets the highlight update perform the scroll", () => {
    const { app, calls } = makeApp(true);
    app.onAudioPlay();
    assert.deepEqual(calls, [["highlight", 12.5], ["section", 3]]);
});

test("poll synchronizes the section indicator with the highlighted index", () => {
    const { app, calls } = makeApp(true);
    app.poll();
    assert.deepEqual(calls, [["highlight", 12.5], ["section", 3]]);
});

test("zombie checks do not trigger resume scrolling", () => {
    const { app, calls } = makeApp();
    app.zombieChecker.isChecking = true;
    app.onAudioPlay();
    assert.deepEqual(calls, []);
});
