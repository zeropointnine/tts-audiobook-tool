"use strict";

/**
 * Display formatting for ABR text segment leaves.
 *
 * A leaf may carry an optional `fmt`: its own `text` plus the tags <h> <b> <i> <u>,
 * with literal &, < and > escaped as &amp; &lt; &gt;. Markup is self-contained per
 * leaf. It applies only when its stripped text equals the leaf text exactly;
 * anything else is ignored and the leaf renders as plain text.
 *
 * Mirrors tts_audiobook_tool/text_ops/text_format.py.
 * See docs-dev/text-formatting-layer.md
 */
class TextFormat {

    static BOLD = 1;
    static ITALIC = 2;
    static UNDERLINE = 4;
    static HEADING = 8;

    static STYLE_BY_TAG = { h: 8, b: 1, i: 2, u: 4 };
    static CLASS_BY_STYLE = [
        [8, "fmtHeading"],
        [1, "fmtBold"],
        [2, "fmtItalic"],
        [4, "fmtUnderline"],
    ];
    static ENTITIES = { amp: "&", lt: "<", gt: ">" };

    /**
     * @param {string} fmt
     * @returns {{text: string, masks: number[]}|null} null when malformed
     */
    static parse(fmt) {
        if (typeof fmt !== "string") {
            return null;
        }
        const regex = /<(\/?)([a-z]*)>|&(amp|lt|gt);|[<>&]/g;
        const parts = [];
        const masks = [];
        const stack = [];
        let current = 0;
        let pos = 0;
        const addLiteral = (literal) => {
            parts.push(literal);
            for (let i = 0; i < literal.length; i++) {
                masks.push(current);
            }
        };

        let match;
        while ((match = regex.exec(fmt)) !== null) {
            addLiteral(fmt.slice(pos, match.index));
            pos = match.index + match[0].length;

            const tag = match[2];
            const entity = match[3];
            if (tag !== undefined) {
                const style = TextFormat.STYLE_BY_TAG[tag];
                if (!style) {
                    return null;
                }
                if (match[1]) {
                    if (stack.length === 0 || stack.at(-1) !== style) {
                        return null;
                    }
                    stack.pop();
                    current &= ~style;
                } else {
                    if (current & style) {
                        return null;
                    }
                    stack.push(style);
                    current |= style;
                }
            } else if (entity !== undefined) {
                parts.push(TextFormat.ENTITIES[entity]);
                masks.push(current);
            } else {
                return null;
            }
        }
        addLiteral(fmt.slice(pos));
        if (stack.length > 0) {
            return null;
        }
        return { text: parts.join(""), masks };
    }

    /**
     * Per-character style masks (parallel to `text`) when `fmt` is valid markup of
     * `text` and styles at least one character; otherwise null.
     */
    static masksFor(fmt, text) {
        if (typeof fmt !== "string" || typeof text !== "string") {
            return null;
        }
        const parsed = TextFormat.parse(fmt);
        if (!parsed || parsed.text !== text || !parsed.masks.some(Boolean)) {
            return null;
        }
        return parsed.masks;
    }

    static classNameFor(mask) {
        return TextFormat.CLASS_BY_STYLE
            .filter(([style]) => mask & style)
            .map(([, className]) => className)
            .join(" ");
    }

    /** Mask of the first or last visible (non-whitespace) character, or 0. */
    static edgeVisibleMask(text, masks, fromEnd) {
        if (!masks) {
            return 0;
        }
        for (let n = 0; n < text.length; n++) {
            const i = fromEnd ? text.length - 1 - n : n;
            if (!/\s/.test(text[i])) {
                return masks[i];
            }
        }
        return 0;
    }

    /**
     * Appends `text` to `parent`, wrapping styled runs in spans.
     * `masks` is parallel to `text`, or null for plain text.
     */
    static appendStyledText(parent, text, masks) {
        if (!text) {
            return;
        }
        if (!masks || !masks.some(Boolean)) {
            parent.appendChild(document.createTextNode(text));
            return;
        }
        let runStart = 0;
        for (let i = 1; i <= text.length; i++) {
            if (i < text.length && masks[i] === masks[runStart]) {
                continue;
            }
            TextFormat.appendRun(parent, text.slice(runStart, i), masks[runStart]);
            runStart = i;
        }
    }

    static appendRun(parent, text, mask) {
        if (!mask) {
            parent.appendChild(document.createTextNode(text));
            return;
        }
        const span = document.createElement("span");
        span.className = TextFormat.classNameFor(mask);
        span.textContent = text;
        parent.appendChild(span);
    }
}
