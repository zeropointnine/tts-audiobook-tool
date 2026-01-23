import js from "@eslint/js";
import globals from "globals";
import json from "@eslint/json";
import css from "@eslint/css";
import { defineConfig } from "eslint/config";


export default defineConfig([

  // App
  {
    files: ["browser_player/app.js"],
    rules: { "no-unused-vars": "off" },
    plugins: { js },
    extends: ["js/recommended"],

    languageOptions: {
      globals: {
        RootAttributer: true,
        MetadataUtil: true,
        DragDropController: true,
        OverlayManager: true,
        StorageController: true,
        SleepTimer: true,
        ZombieChecker: true,
        PlayerVisibilityController: true,

        AudioPlayer: true,
        Bookmarks: true,
        BookText: true,
        Header: true,
        Toast: true,
        Menu: true,

        // External library
        pako: true
      }
    }
  },

  // OverlayManager
  {
    files: ["browser_player/overlay-manager.js"],
    rules: { "no-unused-vars": "off" },
    plugins: { js },
    extends: ["js/recommended"],

    languageOptions: {
      globals: {
        Menu: true
      }
    }
  },

  // Catch-all - applies to all js files
  {
    files: ["**/*.{js,mjs,cjs}"],
    rules: { "no-unused-vars": "off" },
    plugins: { js },
    extends: ["js/recommended"],
    languageOptions: { 
      globals: {
        ...globals.browser,
        // Low-level, oft-used static util classes
        AppUtil: true,
        ShowUtil: true,
        RootAttributer: true,
        Util: true
      }
    } 
  },
  { files: ["**/*.js"], languageOptions: { sourceType: "script" } },

  // Other file types
  { files: ["**/*.json"], plugins: { json }, language: "json/json", extends: ["json/recommended"] },
  { files: ["**/*.css"], plugins: { css }, language: "css/css", extends: ["css/recommended"] }
]);
