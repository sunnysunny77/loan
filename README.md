<!DOCTYPE html>

<html lang="en">
<head><meta charset="utf-8"/>
<meta content="width=device-width, initial-scale=1.0" name="viewport"/>
<title>loan</title><script src="https://cdnjs.cloudflare.com/ajax/libs/require.js/2.1.10/require.min.js"></script>
<style type="text/css">
    pre { line-height: 125%; }
td.linenos .normal { color: inherit; background-color: transparent; padding-left: 5px; padding-right: 5px; }
span.linenos { color: inherit; background-color: transparent; padding-left: 5px; padding-right: 5px; }
td.linenos .special { color: #000000; background-color: #ffffc0; padding-left: 5px; padding-right: 5px; }
span.linenos.special { color: #000000; background-color: #ffffc0; padding-left: 5px; padding-right: 5px; }
.highlight .hll { background-color: var(--jp-cell-editor-active-background) }
.highlight { background: var(--jp-cell-editor-background); color: var(--jp-mirror-editor-variable-color) }
.highlight .c { color: var(--jp-mirror-editor-comment-color); font-style: italic } /* Comment */
.highlight .err { color: var(--jp-mirror-editor-error-color) } /* Error */
.highlight .k { color: var(--jp-mirror-editor-keyword-color); font-weight: bold } /* Keyword */
.highlight .o { color: var(--jp-mirror-editor-operator-color); font-weight: bold } /* Operator */
.highlight .p { color: var(--jp-mirror-editor-punctuation-color) } /* Punctuation */
.highlight .ch { color: var(--jp-mirror-editor-comment-color); font-style: italic } /* Comment.Hashbang */
.highlight .cm { color: var(--jp-mirror-editor-comment-color); font-style: italic } /* Comment.Multiline */
.highlight .cp { color: var(--jp-mirror-editor-comment-color); font-style: italic } /* Comment.Preproc */
.highlight .cpf { color: var(--jp-mirror-editor-comment-color); font-style: italic } /* Comment.PreprocFile */
.highlight .c1 { color: var(--jp-mirror-editor-comment-color); font-style: italic } /* Comment.Single */
.highlight .cs { color: var(--jp-mirror-editor-comment-color); font-style: italic } /* Comment.Special */
.highlight .kc { color: var(--jp-mirror-editor-keyword-color); font-weight: bold } /* Keyword.Constant */
.highlight .kd { color: var(--jp-mirror-editor-keyword-color); font-weight: bold } /* Keyword.Declaration */
.highlight .kn { color: var(--jp-mirror-editor-keyword-color); font-weight: bold } /* Keyword.Namespace */
.highlight .kp { color: var(--jp-mirror-editor-keyword-color); font-weight: bold } /* Keyword.Pseudo */
.highlight .kr { color: var(--jp-mirror-editor-keyword-color); font-weight: bold } /* Keyword.Reserved */
.highlight .kt { color: var(--jp-mirror-editor-keyword-color); font-weight: bold } /* Keyword.Type */
.highlight .m { color: var(--jp-mirror-editor-number-color) } /* Literal.Number */
.highlight .s { color: var(--jp-mirror-editor-string-color) } /* Literal.String */
.highlight .ow { color: var(--jp-mirror-editor-operator-color); font-weight: bold } /* Operator.Word */
.highlight .pm { color: var(--jp-mirror-editor-punctuation-color) } /* Punctuation.Marker */
.highlight .w { color: var(--jp-mirror-editor-variable-color) } /* Text.Whitespace */
.highlight .mb { color: var(--jp-mirror-editor-number-color) } /* Literal.Number.Bin */
.highlight .mf { color: var(--jp-mirror-editor-number-color) } /* Literal.Number.Float */
.highlight .mh { color: var(--jp-mirror-editor-number-color) } /* Literal.Number.Hex */
.highlight .mi { color: var(--jp-mirror-editor-number-color) } /* Literal.Number.Integer */
.highlight .mo { color: var(--jp-mirror-editor-number-color) } /* Literal.Number.Oct */
.highlight .sa { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Affix */
.highlight .sb { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Backtick */
.highlight .sc { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Char */
.highlight .dl { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Delimiter */
.highlight .sd { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Doc */
.highlight .s2 { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Double */
.highlight .se { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Escape */
.highlight .sh { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Heredoc */
.highlight .si { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Interpol */
.highlight .sx { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Other */
.highlight .sr { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Regex */
.highlight .s1 { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Single */
.highlight .ss { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Symbol */
.highlight .il { color: var(--jp-mirror-editor-number-color) } /* Literal.Number.Integer.Long */
  </style>
<style type="text/css">
/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*
 * Mozilla scrollbar styling
 */

/* use standard opaque scrollbars for most nodes */
[data-jp-theme-scrollbars='true'] {
  scrollbar-color: rgb(var(--jp-scrollbar-thumb-color))
    var(--jp-scrollbar-background-color);
}

/* for code nodes, use a transparent style of scrollbar. These selectors
 * will match lower in the tree, and so will override the above */
[data-jp-theme-scrollbars='true'] .CodeMirror-hscrollbar,
[data-jp-theme-scrollbars='true'] .CodeMirror-vscrollbar {
  scrollbar-color: rgba(var(--jp-scrollbar-thumb-color), 0.5) transparent;
}

/* tiny scrollbar */

.jp-scrollbar-tiny {
  scrollbar-color: rgba(var(--jp-scrollbar-thumb-color), 0.5) transparent;
  scrollbar-width: thin;
}

/* tiny scrollbar */

.jp-scrollbar-tiny::-webkit-scrollbar,
.jp-scrollbar-tiny::-webkit-scrollbar-corner {
  background-color: transparent;
  height: 4px;
  width: 4px;
}

.jp-scrollbar-tiny::-webkit-scrollbar-thumb {
  background: rgba(var(--jp-scrollbar-thumb-color), 0.5);
}

.jp-scrollbar-tiny::-webkit-scrollbar-track:horizontal {
  border-left: 0 solid transparent;
  border-right: 0 solid transparent;
}

.jp-scrollbar-tiny::-webkit-scrollbar-track:vertical {
  border-top: 0 solid transparent;
  border-bottom: 0 solid transparent;
}

/*
 * Lumino
 */

.lm-ScrollBar[data-orientation='horizontal'] {
  min-height: 16px;
  max-height: 16px;
  min-width: 45px;
  border-top: 1px solid #a0a0a0;
}

.lm-ScrollBar[data-orientation='vertical'] {
  min-width: 16px;
  max-width: 16px;
  min-height: 45px;
  border-left: 1px solid #a0a0a0;
}

.lm-ScrollBar-button {
  background-color: #f0f0f0;
  background-position: center center;
  min-height: 15px;
  max-height: 15px;
  min-width: 15px;
  max-width: 15px;
}

.lm-ScrollBar-button:hover {
  background-color: #dadada;
}

.lm-ScrollBar-button.lm-mod-active {
  background-color: #cdcdcd;
}

.lm-ScrollBar-track {
  background: #f0f0f0;
}

.lm-ScrollBar-thumb {
  background: #cdcdcd;
}

.lm-ScrollBar-thumb:hover {
  background: #bababa;
}

.lm-ScrollBar-thumb.lm-mod-active {
  background: #a0a0a0;
}

.lm-ScrollBar[data-orientation='horizontal'] .lm-ScrollBar-thumb {
  height: 100%;
  min-width: 15px;
  border-left: 1px solid #a0a0a0;
  border-right: 1px solid #a0a0a0;
}

.lm-ScrollBar[data-orientation='vertical'] .lm-ScrollBar-thumb {
  width: 100%;
  min-height: 15px;
  border-top: 1px solid #a0a0a0;
  border-bottom: 1px solid #a0a0a0;
}

.lm-ScrollBar[data-orientation='horizontal']
  .lm-ScrollBar-button[data-action='decrement'] {
  background-image: var(--jp-icon-caret-left);
  background-size: 17px;
}

.lm-ScrollBar[data-orientation='horizontal']
  .lm-ScrollBar-button[data-action='increment'] {
  background-image: var(--jp-icon-caret-right);
  background-size: 17px;
}

.lm-ScrollBar[data-orientation='vertical']
  .lm-ScrollBar-button[data-action='decrement'] {
  background-image: var(--jp-icon-caret-up);
  background-size: 17px;
}

.lm-ScrollBar[data-orientation='vertical']
  .lm-ScrollBar-button[data-action='increment'] {
  background-image: var(--jp-icon-caret-down);
  background-size: 17px;
}

/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Copyright (c) 2014-2017, PhosphorJS Contributors
|
| Distributed under the terms of the BSD 3-Clause License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/

.lm-Widget {
  box-sizing: border-box;
  position: relative;
  overflow: hidden;
}

.lm-Widget.lm-mod-hidden {
  display: none !important;
}

/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

.lm-AccordionPanel[data-orientation='horizontal'] > .lm-AccordionPanel-title {
  /* Title is rotated for horizontal accordion panel using CSS */
  display: block;
  transform-origin: top left;
  transform: rotate(-90deg) translate(-100%);
}

/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Copyright (c) 2014-2017, PhosphorJS Contributors
|
| Distributed under the terms of the BSD 3-Clause License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/

.lm-CommandPalette {
  display: flex;
  flex-direction: column;
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
}

.lm-CommandPalette-search {
  flex: 0 0 auto;
}

.lm-CommandPalette-content {
  flex: 1 1 auto;
  margin: 0;
  padding: 0;
  min-height: 0;
  overflow: auto;
  list-style-type: none;
}

.lm-CommandPalette-header {
  overflow: hidden;
  white-space: nowrap;
  text-overflow: ellipsis;
}

.lm-CommandPalette-item {
  display: flex;
  flex-direction: row;
}

.lm-CommandPalette-itemIcon {
  flex: 0 0 auto;
}

.lm-CommandPalette-itemContent {
  flex: 1 1 auto;
  overflow: hidden;
}

.lm-CommandPalette-itemShortcut {
  flex: 0 0 auto;
}

.lm-CommandPalette-itemLabel {
  overflow: hidden;
  white-space: nowrap;
  text-overflow: ellipsis;
}

.lm-close-icon {
  border: 1px solid transparent;
  background-color: transparent;
  position: absolute;
  z-index: 1;
  right: 3%;
  top: 0;
  bottom: 0;
  margin: auto;
  padding: 7px 0;
  display: none;
  vertical-align: middle;
  outline: 0;
  cursor: pointer;
}
.lm-close-icon:after {
  content: 'X';
  display: block;
  width: 15px;
  height: 15px;
  text-align: center;
  color: #000;
  font-weight: normal;
  font-size: 12px;
  cursor: pointer;
}

/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Copyright (c) 2014-2017, PhosphorJS Contributors
|
| Distributed under the terms of the BSD 3-Clause License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/

.lm-DockPanel {
  z-index: 0;
}

.lm-DockPanel-widget {
  z-index: 0;
}

.lm-DockPanel-tabBar {
  z-index: 1;
}

.lm-DockPanel-handle {
  z-index: 2;
}

.lm-DockPanel-handle.lm-mod-hidden {
  display: none !important;
}

.lm-DockPanel-handle:after {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  content: '';
}

.lm-DockPanel-handle[data-orientation='horizontal'] {
  cursor: ew-resize;
}

.lm-DockPanel-handle[data-orientation='vertical'] {
  cursor: ns-resize;
}

.lm-DockPanel-handle[data-orientation='horizontal']:after {
  left: 50%;
  min-width: 8px;
  transform: translateX(-50%);
}

.lm-DockPanel-handle[data-orientation='vertical']:after {
  top: 50%;
  min-height: 8px;
  transform: translateY(-50%);
}

.lm-DockPanel-overlay {
  z-index: 3;
  box-sizing: border-box;
  pointer-events: none;
}

.lm-DockPanel-overlay.lm-mod-hidden {
  display: none !important;
}

/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Copyright (c) 2014-2017, PhosphorJS Contributors
|
| Distributed under the terms of the BSD 3-Clause License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/

.lm-Menu {
  z-index: 10000;
  position: absolute;
  white-space: nowrap;
  overflow-x: hidden;
  overflow-y: auto;
  outline: none;
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
}

.lm-Menu-content {
  margin: 0;
  padding: 0;
  display: table;
  list-style-type: none;
}

.lm-Menu-item {
  display: table-row;
}

.lm-Menu-item.lm-mod-hidden,
.lm-Menu-item.lm-mod-collapsed {
  display: none !important;
}

.lm-Menu-itemIcon,
.lm-Menu-itemSubmenuIcon {
  display: table-cell;
  text-align: center;
}

.lm-Menu-itemLabel {
  display: table-cell;
  text-align: left;
}

.lm-Menu-itemShortcut {
  display: table-cell;
  text-align: right;
}

/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Copyright (c) 2014-2017, PhosphorJS Contributors
|
| Distributed under the terms of the BSD 3-Clause License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/

.lm-MenuBar {
  outline: none;
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
}

.lm-MenuBar-content {
  margin: 0;
  padding: 0;
  display: flex;
  flex-direction: row;
  list-style-type: none;
}

.lm-MenuBar-item {
  box-sizing: border-box;
}

.lm-MenuBar-itemIcon,
.lm-MenuBar-itemLabel {
  display: inline-block;
}

/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Copyright (c) 2014-2017, PhosphorJS Contributors
|
| Distributed under the terms of the BSD 3-Clause License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/

.lm-ScrollBar {
  display: flex;
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
}

.lm-ScrollBar[data-orientation='horizontal'] {
  flex-direction: row;
}

.lm-ScrollBar[data-orientation='vertical'] {
  flex-direction: column;
}

.lm-ScrollBar-button {
  box-sizing: border-box;
  flex: 0 0 auto;
}

.lm-ScrollBar-track {
  box-sizing: border-box;
  position: relative;
  overflow: hidden;
  flex: 1 1 auto;
}

.lm-ScrollBar-thumb {
  box-sizing: border-box;
  position: absolute;
}

/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Copyright (c) 2014-2017, PhosphorJS Contributors
|
| Distributed under the terms of the BSD 3-Clause License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/

.lm-SplitPanel-child {
  z-index: 0;
}

.lm-SplitPanel-handle {
  z-index: 1;
}

.lm-SplitPanel-handle.lm-mod-hidden {
  display: none !important;
}

.lm-SplitPanel-handle:after {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  content: '';
}

.lm-SplitPanel[data-orientation='horizontal'] > .lm-SplitPanel-handle {
  cursor: ew-resize;
}

.lm-SplitPanel[data-orientation='vertical'] > .lm-SplitPanel-handle {
  cursor: ns-resize;
}

.lm-SplitPanel[data-orientation='horizontal'] > .lm-SplitPanel-handle:after {
  left: 50%;
  min-width: 8px;
  transform: translateX(-50%);
}

.lm-SplitPanel[data-orientation='vertical'] > .lm-SplitPanel-handle:after {
  top: 50%;
  min-height: 8px;
  transform: translateY(-50%);
}

/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Copyright (c) 2014-2017, PhosphorJS Contributors
|
| Distributed under the terms of the BSD 3-Clause License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/

.lm-TabBar {
  display: flex;
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
}

.lm-TabBar[data-orientation='horizontal'] {
  flex-direction: row;
  align-items: flex-end;
}

.lm-TabBar[data-orientation='vertical'] {
  flex-direction: column;
  align-items: flex-end;
}

.lm-TabBar-content {
  margin: 0;
  padding: 0;
  display: flex;
  flex: 1 1 auto;
  list-style-type: none;
}

.lm-TabBar[data-orientation='horizontal'] > .lm-TabBar-content {
  flex-direction: row;
}

.lm-TabBar[data-orientation='vertical'] > .lm-TabBar-content {
  flex-direction: column;
}

.lm-TabBar-tab {
  display: flex;
  flex-direction: row;
  box-sizing: border-box;
  overflow: hidden;
  touch-action: none; /* Disable native Drag/Drop */
}

.lm-TabBar-tabIcon,
.lm-TabBar-tabCloseIcon {
  flex: 0 0 auto;
}

.lm-TabBar-tabLabel {
  flex: 1 1 auto;
  overflow: hidden;
  white-space: nowrap;
}

.lm-TabBar-tabInput {
  user-select: all;
  width: 100%;
  box-sizing: border-box;
}

.lm-TabBar-tab.lm-mod-hidden {
  display: none !important;
}

.lm-TabBar-addButton.lm-mod-hidden {
  display: none !important;
}

.lm-TabBar.lm-mod-dragging .lm-TabBar-tab {
  position: relative;
}

.lm-TabBar.lm-mod-dragging[data-orientation='horizontal'] .lm-TabBar-tab {
  left: 0;
  transition: left 150ms ease;
}

.lm-TabBar.lm-mod-dragging[data-orientation='vertical'] .lm-TabBar-tab {
  top: 0;
  transition: top 150ms ease;
}

.lm-TabBar.lm-mod-dragging .lm-TabBar-tab.lm-mod-dragging {
  transition: none;
}

.lm-TabBar-tabLabel .lm-TabBar-tabInput {
  user-select: all;
  width: 100%;
  box-sizing: border-box;
  background: inherit;
}

/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Copyright (c) 2014-2017, PhosphorJS Contributors
|
| Distributed under the terms of the BSD 3-Clause License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/

.lm-TabPanel-tabBar {
  z-index: 1;
}

.lm-TabPanel-stackedPanel {
  z-index: 0;
}

/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Copyright (c) 2014-2017, PhosphorJS Contributors
|
| Distributed under the terms of the BSD 3-Clause License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-Collapse {
  display: flex;
  flex-direction: column;
  align-items: stretch;
}

.jp-Collapse-header {
  padding: 1px 12px;
  background-color: var(--jp-layout-color1);
  border-bottom: solid var(--jp-border-width) var(--jp-border-color2);
  color: var(--jp-ui-font-color1);
  cursor: pointer;
  display: flex;
  align-items: center;
  font-size: var(--jp-ui-font-size0);
  font-weight: 600;
  text-transform: uppercase;
  user-select: none;
}

.jp-Collapser-icon {
  height: 16px;
}

.jp-Collapse-header-collapsed .jp-Collapser-icon {
  transform: rotate(-90deg);
  margin: auto 0;
}

.jp-Collapser-title {
  line-height: 25px;
}

.jp-Collapse-contents {
  padding: 0 12px;
  background-color: var(--jp-layout-color1);
  color: var(--jp-ui-font-color1);
  overflow: auto;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/* This file was auto-generated by ensureUiComponents() in @jupyterlab/buildutils */

/**
 * (DEPRECATED) Support for consuming icons as CSS background images
 */

/* Icons urls */

:root {
  --jp-icon-add-above: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTQiIGhlaWdodD0iMTQiIHZpZXdCb3g9IjAgMCAxNCAxNCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPGcgY2xpcC1wYXRoPSJ1cmwoI2NsaXAwXzEzN18xOTQ5MikiPgo8cGF0aCBjbGFzcz0ianAtaWNvbjMiIGQ9Ik00Ljc1IDQuOTMwNjZINi42MjVWNi44MDU2NkM2LjYyNSA3LjAxMTkxIDYuNzkzNzUgNy4xODA2NiA3IDcuMTgwNjZDNy4yMDYyNSA3LjE4MDY2IDcuMzc1IDcuMDExOTEgNy4zNzUgNi44MDU2NlY0LjkzMDY2SDkuMjVDOS40NTYyNSA0LjkzMDY2IDkuNjI1IDQuNzYxOTEgOS42MjUgNC41NTU2NkM5LjYyNSA0LjM0OTQxIDkuNDU2MjUgNC4xODA2NiA5LjI1IDQuMTgwNjZINy4zNzVWMi4zMDU2NkM3LjM3NSAyLjA5OTQxIDcuMjA2MjUgMS45MzA2NiA3IDEuOTMwNjZDNi43OTM3NSAxLjkzMDY2IDYuNjI1IDIuMDk5NDEgNi42MjUgMi4zMDU2NlY0LjE4MDY2SDQuNzVDNC41NDM3NSA0LjE4MDY2IDQuMzc1IDQuMzQ5NDEgNC4zNzUgNC41NTU2NkM0LjM3NSA0Ljc2MTkxIDQuNTQzNzUgNC45MzA2NiA0Ljc1IDQuOTMwNjZaIiBmaWxsPSIjNjE2MTYxIiBzdHJva2U9IiM2MTYxNjEiIHN0cm9rZS13aWR0aD0iMC43Ii8+CjwvZz4KPHBhdGggY2xhc3M9ImpwLWljb24zIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiIGNsaXAtcnVsZT0iZXZlbm9kZCIgZD0iTTExLjUgOS41VjExLjVMMi41IDExLjVWOS41TDExLjUgOS41Wk0xMiA4QzEyLjU1MjMgOCAxMyA4LjQ0NzcyIDEzIDlWMTJDMTMgMTIuNTUyMyAxMi41NTIzIDEzIDEyIDEzTDIgMTNDMS40NDc3MiAxMyAxIDEyLjU1MjMgMSAxMlY5QzEgOC40NDc3MiAxLjQ0NzcxIDggMiA4TDEyIDhaIiBmaWxsPSIjNjE2MTYxIi8+CjxkZWZzPgo8Y2xpcFBhdGggaWQ9ImNsaXAwXzEzN18xOTQ5MiI+CjxyZWN0IGNsYXNzPSJqcC1pY29uMyIgd2lkdGg9IjYiIGhlaWdodD0iNiIgZmlsbD0id2hpdGUiIHRyYW5zZm9ybT0ibWF0cml4KC0xIDAgMCAxIDEwIDEuNTU1NjYpIi8+CjwvY2xpcFBhdGg+CjwvZGVmcz4KPC9zdmc+Cg==);
  --jp-icon-add-below: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTQiIGhlaWdodD0iMTQiIHZpZXdCb3g9IjAgMCAxNCAxNCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPGcgY2xpcC1wYXRoPSJ1cmwoI2NsaXAwXzEzN18xOTQ5OCkiPgo8cGF0aCBjbGFzcz0ianAtaWNvbjMiIGQ9Ik05LjI1IDEwLjA2OTNMNy4zNzUgMTAuMDY5M0w3LjM3NSA4LjE5NDM0QzcuMzc1IDcuOTg4MDkgNy4yMDYyNSA3LjgxOTM0IDcgNy44MTkzNEM2Ljc5Mzc1IDcuODE5MzQgNi42MjUgNy45ODgwOSA2LjYyNSA4LjE5NDM0TDYuNjI1IDEwLjA2OTNMNC43NSAxMC4wNjkzQzQuNTQzNzUgMTAuMDY5MyA0LjM3NSAxMC4yMzgxIDQuMzc1IDEwLjQ0NDNDNC4zNzUgMTAuNjUwNiA0LjU0Mzc1IDEwLjgxOTMgNC43NSAxMC44MTkzTDYuNjI1IDEwLjgxOTNMNi42MjUgMTIuNjk0M0M2LjYyNSAxMi45MDA2IDYuNzkzNzUgMTMuMDY5MyA3IDEzLjA2OTNDNy4yMDYyNSAxMy4wNjkzIDcuMzc1IDEyLjkwMDYgNy4zNzUgMTIuNjk0M0w3LjM3NSAxMC44MTkzTDkuMjUgMTAuODE5M0M5LjQ1NjI1IDEwLjgxOTMgOS42MjUgMTAuNjUwNiA5LjYyNSAxMC40NDQzQzkuNjI1IDEwLjIzODEgOS40NTYyNSAxMC4wNjkzIDkuMjUgMTAuMDY5M1oiIGZpbGw9IiM2MTYxNjEiIHN0cm9rZT0iIzYxNjE2MSIgc3Ryb2tlLXdpZHRoPSIwLjciLz4KPC9nPgo8cGF0aCBjbGFzcz0ianAtaWNvbjMiIGZpbGwtcnVsZT0iZXZlbm9kZCIgY2xpcC1ydWxlPSJldmVub2RkIiBkPSJNMi41IDUuNUwyLjUgMy41TDExLjUgMy41TDExLjUgNS41TDIuNSA1LjVaTTIgN0MxLjQ0NzcyIDcgMSA2LjU1MjI4IDEgNkwxIDNDMSAyLjQ0NzcyIDEuNDQ3NzIgMiAyIDJMMTIgMkMxMi41NTIzIDIgMTMgMi40NDc3MiAxMyAzTDEzIDZDMTMgNi41NTIyOSAxMi41NTIzIDcgMTIgN0wyIDdaIiBmaWxsPSIjNjE2MTYxIi8+CjxkZWZzPgo8Y2xpcFBhdGggaWQ9ImNsaXAwXzEzN18xOTQ5OCI+CjxyZWN0IGNsYXNzPSJqcC1pY29uMyIgd2lkdGg9IjYiIGhlaWdodD0iNiIgZmlsbD0id2hpdGUiIHRyYW5zZm9ybT0ibWF0cml4KDEgMS43NDg0NmUtMDcgMS43NDg0NmUtMDcgLTEgNCAxMy40NDQzKSIvPgo8L2NsaXBQYXRoPgo8L2RlZnM+Cjwvc3ZnPgo=);
  --jp-icon-add: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTE5IDEzaC02djZoLTJ2LTZINXYtMmg2VjVoMnY2aDZ2MnoiLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-bell: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDE2IDE2IiB2ZXJzaW9uPSIxLjEiPgogICA8cGF0aCBjbGFzcz0ianAtaWNvbjIganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjMzMzMzMzIgogICAgICBkPSJtOCAwLjI5Yy0xLjQgMC0yLjcgMC43My0zLjYgMS44LTEuMiAxLjUtMS40IDMuNC0xLjUgNS4yLTAuMTggMi4yLTAuNDQgNC0yLjMgNS4zbDAuMjggMS4zaDVjMC4wMjYgMC42NiAwLjMyIDEuMSAwLjcxIDEuNSAwLjg0IDAuNjEgMiAwLjYxIDIuOCAwIDAuNTItMC40IDAuNi0xIDAuNzEtMS41aDVsMC4yOC0xLjNjLTEuOS0wLjk3LTIuMi0zLjMtMi4zLTUuMy0wLjEzLTEuOC0wLjI2LTMuNy0xLjUtNS4yLTAuODUtMS0yLjItMS44LTMuNi0xLjh6bTAgMS40YzAuODggMCAxLjkgMC41NSAyLjUgMS4zIDAuODggMS4xIDEuMSAyLjcgMS4yIDQuNCAwLjEzIDEuNyAwLjIzIDMuNiAxLjMgNS4yaC0xMGMxLjEtMS42IDEuMi0zLjQgMS4zLTUuMiAwLjEzLTEuNyAwLjMtMy4zIDEuMi00LjQgMC41OS0wLjcyIDEuNi0xLjMgMi41LTEuM3ptLTAuNzQgMTJoMS41Yy0wLjAwMTUgMC4yOCAwLjAxNSAwLjc5LTAuNzQgMC43OS0wLjczIDAuMDAxNi0wLjcyLTAuNTMtMC43NC0wLjc5eiIgLz4KPC9zdmc+Cg==);
  --jp-icon-bug-dot: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjQiIGhlaWdodD0iMjQiIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICAgIDxnIGNsYXNzPSJqcC1pY29uMyBqcC1pY29uLXNlbGVjdGFibGUiIGZpbGw9IiM2MTYxNjEiPgogICAgICAgIDxwYXRoIGZpbGwtcnVsZT0iZXZlbm9kZCIgY2xpcC1ydWxlPSJldmVub2RkIiBkPSJNMTcuMTkgOEgyMFYxMEgxNy45MUMxNy45NiAxMC4zMyAxOCAxMC42NiAxOCAxMVYxMkgyMFYxNEgxOC41SDE4VjE0LjAyNzVDMTUuNzUgMTQuMjc2MiAxNCAxNi4xODM3IDE0IDE4LjVDMTQgMTkuMjA4IDE0LjE2MzUgMTkuODc3OSAxNC40NTQ5IDIwLjQ3MzlDMTMuNzA2MyAyMC44MTE3IDEyLjg3NTcgMjEgMTIgMjFDOS43OCAyMSA3Ljg1IDE5Ljc5IDYuODEgMThINFYxNkg2LjA5QzYuMDQgMTUuNjcgNiAxNS4zNCA2IDE1VjE0SDRWMTJINlYxMUM2IDEwLjY2IDYuMDQgMTAuMzMgNi4wOSAxMEg0VjhINi44MUM3LjI2IDcuMjIgNy44OCA2LjU1IDguNjIgNi4wNEw3IDQuNDFMOC40MSAzTDEwLjU5IDUuMTdDMTEuMDQgNS4wNiAxMS41MSA1IDEyIDVDMTIuNDkgNSAxMi45NiA1LjA2IDEzLjQyIDUuMTdMMTUuNTkgM0wxNyA0LjQxTDE1LjM3IDYuMDRDMTYuMTIgNi41NSAxNi43NCA3LjIyIDE3LjE5IDhaTTEwIDE2SDE0VjE0SDEwVjE2Wk0xMCAxMkgxNFYxMEgxMFYxMloiIGZpbGw9IiM2MTYxNjEiLz4KICAgICAgICA8cGF0aCBkPSJNMjIgMTguNUMyMiAyMC40MzMgMjAuNDMzIDIyIDE4LjUgMjJDMTYuNTY3IDIyIDE1IDIwLjQzMyAxNSAxOC41QzE1IDE2LjU2NyAxNi41NjcgMTUgMTguNSAxNUMyMC40MzMgMTUgMjIgMTYuNTY3IDIyIDE4LjVaIiBmaWxsPSIjNjE2MTYxIi8+CiAgICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-bug: url(data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIxNiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjNjE2MTYxIj4KICAgIDxwYXRoIGQ9Ik0yMCA4aC0yLjgxYy0uNDUtLjc4LTEuMDctMS40NS0xLjgyLTEuOTZMMTcgNC40MSAxNS41OSAzbC0yLjE3IDIuMTdDMTIuOTYgNS4wNiAxMi40OSA1IDEyIDVjLS40OSAwLS45Ni4wNi0xLjQxLjE3TDguNDEgMyA3IDQuNDFsMS42MiAxLjYzQzcuODggNi41NSA3LjI2IDcuMjIgNi44MSA4SDR2MmgyLjA5Yy0uMDUuMzMtLjA5LjY2LS4wOSAxdjFINHYyaDJ2MWMwIC4zNC4wNC42Ny4wOSAxSDR2MmgyLjgxYzEuMDQgMS43OSAyLjk3IDMgNS4xOSAzczQuMTUtMS4yMSA1LjE5LTNIMjB2LTJoLTIuMDljLjA1LS4zMy4wOS0uNjYuMDktMXYtMWgydi0yaC0ydi0xYzAtLjM0LS4wNC0uNjctLjA5LTFIMjBWOHptLTYgOGgtNHYtMmg0djJ6bTAtNGgtNHYtMmg0djJ6Ii8+CiAgPC9nPgo8L3N2Zz4K);
  --jp-icon-build: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTYiIHZpZXdCb3g9IjAgMCAyNCAyNCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTE0LjkgMTcuNDVDMTYuMjUgMTcuNDUgMTcuMzUgMTYuMzUgMTcuMzUgMTVDMTcuMzUgMTMuNjUgMTYuMjUgMTIuNTUgMTQuOSAxMi41NUMxMy41NCAxMi41NSAxMi40NSAxMy42NSAxMi40NSAxNUMxMi40NSAxNi4zNSAxMy41NCAxNy40NSAxNC45IDE3LjQ1Wk0yMC4xIDE1LjY4TDIxLjU4IDE2Ljg0QzIxLjcxIDE2Ljk1IDIxLjc1IDE3LjEzIDIxLjY2IDE3LjI5TDIwLjI2IDE5LjcxQzIwLjE3IDE5Ljg2IDIwIDE5LjkyIDE5LjgzIDE5Ljg2TDE4LjA5IDE5LjE2QzE3LjczIDE5LjQ0IDE3LjMzIDE5LjY3IDE2LjkxIDE5Ljg1TDE2LjY0IDIxLjdDMTYuNjIgMjEuODcgMTYuNDcgMjIgMTYuMyAyMkgxMy41QzEzLjMyIDIyIDEzLjE4IDIxLjg3IDEzLjE1IDIxLjdMMTIuODkgMTkuODVDMTIuNDYgMTkuNjcgMTIuMDcgMTkuNDQgMTEuNzEgMTkuMTZMOS45NjAwMiAxOS44NkM5LjgxMDAyIDE5LjkyIDkuNjIwMDIgMTkuODYgOS41NDAwMiAxOS43MUw4LjE0MDAyIDE3LjI5QzguMDUwMDIgMTcuMTMgOC4wOTAwMiAxNi45NSA4LjIyMDAyIDE2Ljg0TDkuNzAwMDIgMTUuNjhMOS42NTAwMSAxNUw5LjcwMDAyIDE0LjMxTDguMjIwMDIgMTMuMTZDOC4wOTAwMiAxMy4wNSA4LjA1MDAyIDEyLjg2IDguMTQwMDIgMTIuNzFMOS41NDAwMiAxMC4yOUM5LjYyMDAyIDEwLjEzIDkuODEwMDIgMTAuMDcgOS45NjAwMiAxMC4xM0wxMS43MSAxMC44NEMxMi4wNyAxMC41NiAxMi40NiAxMC4zMiAxMi44OSAxMC4xNUwxMy4xNSA4LjI4OTk4QzEzLjE4IDguMTI5OTggMTMuMzIgNy45OTk5OCAxMy41IDcuOTk5OThIMTYuM0MxNi40NyA3Ljk5OTk4IDE2LjYyIDguMTI5OTggMTYuNjQgOC4yODk5OEwxNi45MSAxMC4xNUMxNy4zMyAxMC4zMiAxNy43MyAxMC41NiAxOC4wOSAxMC44NEwxOS44MyAxMC4xM0MyMCAxMC4wNyAyMC4xNyAxMC4xMyAyMC4yNiAxMC4yOUwyMS42NiAxMi43MUMyMS43NSAxMi44NiAyMS43MSAxMy4wNSAyMS41OCAxMy4xNkwyMC4xIDE0LjMxTDIwLjE1IDE1TDIwLjEgMTUuNjhaIi8+CiAgICA8cGF0aCBkPSJNNy4zMjk2NiA3LjQ0NDU0QzguMDgzMSA3LjAwOTU0IDguMzM5MzIgNi4wNTMzMiA3LjkwNDMyIDUuMjk5ODhDNy40NjkzMiA0LjU0NjQzIDYuNTA4MSA0LjI4MTU2IDUuNzU0NjYgNC43MTY1NkM1LjM5MTc2IDQuOTI2MDggNS4xMjY5NSA1LjI3MTE4IDUuMDE4NDkgNS42NzU5NEM0LjkxMDA0IDYuMDgwNzEgNC45NjY4MiA2LjUxMTk4IDUuMTc2MzQgNi44NzQ4OEM1LjYxMTM0IDcuNjI4MzIgNi41NzYyMiA3Ljg3OTU0IDcuMzI5NjYgNy40NDQ1NFpNOS42NTcxOCA0Ljc5NTkzTDEwLjg2NzIgNC45NTE3OUMxMC45NjI4IDQuOTc3NDEgMTEuMDQwMiA1LjA3MTMzIDExLjAzODIgNS4xODc5M0wxMS4wMzg4IDYuOTg4OTNDMTEuMDQ1NSA3LjEwMDU0IDEwLjk2MTYgNy4xOTUxOCAxMC44NTUgNy4yMTA1NEw5LjY2MDAxIDcuMzgwODNMOS4yMzkxNSA4LjEzMTg4TDkuNjY5NjEgOS4yNTc0NUM5LjcwNzI5IDkuMzYyNzEgOS42NjkzNCA5LjQ3Njk5IDkuNTc0MDggOS41MzE5OUw4LjAxNTIzIDEwLjQzMkM3LjkxMTMxIDEwLjQ5MiA3Ljc5MzM3IDEwLjQ2NzcgNy43MjEwNSAxMC4zODI0TDYuOTg3NDggOS40MzE4OEw2LjEwOTMxIDkuNDMwODNMNS4zNDcwNCAxMC4zOTA1QzUuMjg5MDkgMTAuNDcwMiA1LjE3MzgzIDEwLjQ5MDUgNS4wNzE4NyAxMC40MzM5TDMuNTEyNDUgOS41MzI5M0MzLjQxMDQ5IDkuNDc2MzMgMy4zNzY0NyA5LjM1NzQxIDMuNDEwNzUgOS4yNTY3OUwzLjg2MzQ3IDguMTQwOTNMMy42MTc0OSA3Ljc3NDg4TDMuNDIzNDcgNy4zNzg4M0wyLjIzMDc1IDcuMjEyOTdDMi4xMjY0NyA3LjE5MjM1IDIuMDQwNDkgNy4xMDM0MiAyLjA0MjQ1IDYuOTg2ODJMMi4wNDE4NyA1LjE4NTgyQzIuMDQzODMgNS4wNjkyMiAyLjExOTA5IDQuOTc5NTggMi4yMTcwNCA0Ljk2OTIyTDMuNDIwNjUgNC43OTM5M0wzLjg2NzQ5IDQuMDI3ODhMMy40MTEwNSAyLjkxNzMxQzMuMzczMzcgMi44MTIwNCAzLjQxMTMxIDIuNjk3NzYgMy41MTUyMyAyLjYzNzc2TDUuMDc0MDggMS43Mzc3NkM1LjE2OTM0IDEuNjgyNzYgNS4yODcyOSAxLjcwNzA0IDUuMzU5NjEgMS43OTIzMUw2LjExOTE1IDIuNzI3ODhMNi45ODAwMSAyLjczODkzTDcuNzI0OTYgMS43ODkyMkM3Ljc5MTU2IDEuNzA0NTggNy45MTU0OCAxLjY3OTIyIDguMDA4NzkgMS43NDA4Mkw5LjU2ODIxIDIuNjQxODJDOS42NzAxNyAyLjY5ODQyIDkuNzEyODUgMi44MTIzNCA5LjY4NzIzIDIuOTA3OTdMOS4yMTcxOCA0LjAzMzgzTDkuNDYzMTYgNC4zOTk4OEw5LjY1NzE4IDQuNzk1OTNaIi8+CiAgPC9nPgo8L3N2Zz4K);
  --jp-icon-caret-down-empty-thin: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIwIDIwIj4KCTxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSIgc2hhcGUtcmVuZGVyaW5nPSJnZW9tZXRyaWNQcmVjaXNpb24iPgoJCTxwb2x5Z29uIGNsYXNzPSJzdDEiIHBvaW50cz0iOS45LDEzLjYgMy42LDcuNCA0LjQsNi42IDkuOSwxMi4yIDE1LjQsNi43IDE2LjEsNy40ICIvPgoJPC9nPgo8L3N2Zz4K);
  --jp-icon-caret-down-empty: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDE4IDE4Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiIHNoYXBlLXJlbmRlcmluZz0iZ2VvbWV0cmljUHJlY2lzaW9uIj4KICAgIDxwYXRoIGQ9Ik01LjIsNS45TDksOS43bDMuOC0zLjhsMS4yLDEuMmwtNC45LDVsLTQuOS01TDUuMiw1Ljl6Ii8+CiAgPC9nPgo8L3N2Zz4K);
  --jp-icon-caret-down: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDE4IDE4Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiIHNoYXBlLXJlbmRlcmluZz0iZ2VvbWV0cmljUHJlY2lzaW9uIj4KICAgIDxwYXRoIGQ9Ik01LjIsNy41TDksMTEuMmwzLjgtMy44SDUuMnoiLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-caret-left: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDE4IDE4Ij4KCTxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSIgc2hhcGUtcmVuZGVyaW5nPSJnZW9tZXRyaWNQcmVjaXNpb24iPgoJCTxwYXRoIGQ9Ik0xMC44LDEyLjhMNy4xLDlsMy44LTMuOGwwLDcuNkgxMC44eiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-caret-right: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDE4IDE4Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiIHNoYXBlLXJlbmRlcmluZz0iZ2VvbWV0cmljUHJlY2lzaW9uIj4KICAgIDxwYXRoIGQ9Ik03LjIsNS4yTDEwLjksOWwtMy44LDMuOFY1LjJINy4yeiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-caret-up-empty-thin: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIwIDIwIj4KCTxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSIgc2hhcGUtcmVuZGVyaW5nPSJnZW9tZXRyaWNQcmVjaXNpb24iPgoJCTxwb2x5Z29uIGNsYXNzPSJzdDEiIHBvaW50cz0iMTUuNCwxMy4zIDkuOSw3LjcgNC40LDEzLjIgMy42LDEyLjUgOS45LDYuMyAxNi4xLDEyLjYgIi8+Cgk8L2c+Cjwvc3ZnPgo=);
  --jp-icon-caret-up: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDE4IDE4Ij4KCTxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSIgc2hhcGUtcmVuZGVyaW5nPSJnZW9tZXRyaWNQcmVjaXNpb24iPgoJCTxwYXRoIGQ9Ik01LjIsMTAuNUw5LDYuOGwzLjgsMy44SDUuMnoiLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-case-sensitive: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIwIDIwIj4KICA8ZyBjbGFzcz0ianAtaWNvbjIiIGZpbGw9IiM0MTQxNDEiPgogICAgPHJlY3QgeD0iMiIgeT0iMiIgd2lkdGg9IjE2IiBoZWlnaHQ9IjE2Ii8+CiAgPC9nPgogIDxnIGNsYXNzPSJqcC1pY29uLWFjY2VudDIiIGZpbGw9IiNGRkYiPgogICAgPHBhdGggZD0iTTcuNiw4aDAuOWwzLjUsOGgtMS4xTDEwLDE0SDZsLTAuOSwySDRMNy42LDh6IE04LDkuMUw2LjQsMTNoMy4yTDgsOS4xeiIvPgogICAgPHBhdGggZD0iTTE2LjYsOS44Yy0wLjIsMC4xLTAuNCwwLjEtMC43LDAuMWMtMC4yLDAtMC40LTAuMS0wLjYtMC4yYy0wLjEtMC4xLTAuMi0wLjQtMC4yLTAuNyBjLTAuMywwLjMtMC42LDAuNS0wLjksMC43Yy0wLjMsMC4xLTAuNywwLjItMS4xLDAuMmMtMC4zLDAtMC41LDAtMC43LTAuMWMtMC4yLTAuMS0wLjQtMC4yLTAuNi0wLjNjLTAuMi0wLjEtMC4zLTAuMy0wLjQtMC41IGMtMC4xLTAuMi0wLjEtMC40LTAuMS0wLjdjMC0wLjMsMC4xLTAuNiwwLjItMC44YzAuMS0wLjIsMC4zLTAuNCwwLjQtMC41QzEyLDcsMTIuMiw2LjksMTIuNSw2LjhjMC4yLTAuMSwwLjUtMC4xLDAuNy0wLjIgYzAuMy0wLjEsMC41LTAuMSwwLjctMC4xYzAuMiwwLDAuNC0wLjEsMC42LTAuMWMwLjIsMCwwLjMtMC4xLDAuNC0wLjJjMC4xLTAuMSwwLjItMC4yLDAuMi0wLjRjMC0xLTEuMS0xLTEuMy0xIGMtMC40LDAtMS40LDAtMS40LDEuMmgtMC45YzAtMC40LDAuMS0wLjcsMC4yLTFjMC4xLTAuMiwwLjMtMC40LDAuNS0wLjZjMC4yLTAuMiwwLjUtMC4zLDAuOC0wLjNDMTMuMyw0LDEzLjYsNCwxMy45LDQgYzAuMywwLDAuNSwwLDAuOCwwLjFjMC4zLDAsMC41LDAuMSwwLjcsMC4yYzAuMiwwLjEsMC40LDAuMywwLjUsMC41QzE2LDUsMTYsNS4yLDE2LDUuNnYyLjljMCwwLjIsMCwwLjQsMCwwLjUgYzAsMC4xLDAuMSwwLjIsMC4zLDAuMmMwLjEsMCwwLjIsMCwwLjMsMFY5Ljh6IE0xNS4yLDYuOWMtMS4yLDAuNi0zLjEsMC4yLTMuMSwxLjRjMCwxLjQsMy4xLDEsMy4xLTAuNVY2Ljl6Ii8+CiAgPC9nPgo8L3N2Zz4K);
  --jp-icon-check: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjNjE2MTYxIj4KICAgIDxwYXRoIGQ9Ik05IDE2LjE3TDQuODMgMTJsLTEuNDIgMS40MUw5IDE5IDIxIDdsLTEuNDEtMS40MXoiLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-circle-empty: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTEyIDJDNi40NyAyIDIgNi40NyAyIDEyczQuNDcgMTAgMTAgMTAgMTAtNC40NyAxMC0xMFMxNy41MyAyIDEyIDJ6bTAgMThjLTQuNDEgMC04LTMuNTktOC04czMuNTktOCA4LTggOCAzLjU5IDggOC0zLjU5IDgtOCA4eiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-circle: url(data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMTggMTgiIHdpZHRoPSIxNiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPGNpcmNsZSBjeD0iOSIgY3k9IjkiIHI9IjgiLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-clear: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8bWFzayBpZD0iZG9udXRIb2xlIj4KICAgIDxyZWN0IHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgZmlsbD0id2hpdGUiIC8+CiAgICA8Y2lyY2xlIGN4PSIxMiIgY3k9IjEyIiByPSI4IiBmaWxsPSJibGFjayIvPgogIDwvbWFzaz4KCiAgPGcgY2xhc3M9ImpwLWljb24zIiBmaWxsPSIjNjE2MTYxIj4KICAgIDxyZWN0IGhlaWdodD0iMTgiIHdpZHRoPSIyIiB4PSIxMSIgeT0iMyIgdHJhbnNmb3JtPSJyb3RhdGUoMzE1LCAxMiwgMTIpIi8+CiAgICA8Y2lyY2xlIGN4PSIxMiIgY3k9IjEyIiByPSIxMCIgbWFzaz0idXJsKCNkb251dEhvbGUpIi8+CiAgPC9nPgo8L3N2Zz4K);
  --jp-icon-close: url(data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIxNiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbi1ub25lIGpwLWljb24tc2VsZWN0YWJsZS1pbnZlcnNlIGpwLWljb24zLWhvdmVyIiBmaWxsPSJub25lIj4KICAgIDxjaXJjbGUgY3g9IjEyIiBjeT0iMTIiIHI9IjExIi8+CiAgPC9nPgoKICA8ZyBjbGFzcz0ianAtaWNvbjMganAtaWNvbi1zZWxlY3RhYmxlIGpwLWljb24tYWNjZW50Mi1ob3ZlciIgZmlsbD0iIzYxNjE2MSI+CiAgICA8cGF0aCBkPSJNMTkgNi40MUwxNy41OSA1IDEyIDEwLjU5IDYuNDEgNSA1IDYuNDEgMTAuNTkgMTIgNSAxNy41OSA2LjQxIDE5IDEyIDEzLjQxIDE3LjU5IDE5IDE5IDE3LjU5IDEzLjQxIDEyeiIvPgogIDwvZz4KCiAgPGcgY2xhc3M9ImpwLWljb24tbm9uZSBqcC1pY29uLWJ1c3kiIGZpbGw9Im5vbmUiPgogICAgPGNpcmNsZSBjeD0iMTIiIGN5PSIxMiIgcj0iNyIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-code-check: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjNjE2MTYxIiBzaGFwZS1yZW5kZXJpbmc9Imdlb21ldHJpY1ByZWNpc2lvbiI+CiAgICA8cGF0aCBkPSJNNi41OSwzLjQxTDIsOEw2LjU5LDEyLjZMOCwxMS4xOEw0LjgyLDhMOCw0LjgyTDYuNTksMy40MU0xMi40MSwzLjQxTDExLDQuODJMMTQuMTgsOEwxMSwxMS4xOEwxMi40MSwxMi42TDE3LDhMMTIuNDEsMy40MU0yMS41OSwxMS41OUwxMy41LDE5LjY4TDkuODMsMTZMOC40MiwxNy40MUwxMy41LDIyLjVMMjMsMTNMMjEuNTksMTEuNTlaIiAvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-code: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjIiIGhlaWdodD0iMjIiIHZpZXdCb3g9IjAgMCAyOCAyOCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KCTxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSI+CgkJPHBhdGggZD0iTTExLjQgMTguNkw2LjggMTRMMTEuNCA5LjRMMTAgOEw0IDE0TDEwIDIwTDExLjQgMTguNlpNMTYuNiAxOC42TDIxLjIgMTRMMTYuNiA5LjRMMTggOEwyNCAxNEwxOCAyMEwxNi42IDE4LjZWMTguNloiLz4KCTwvZz4KPC9zdmc+Cg==);
  --jp-icon-collapse-all: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICAgIDxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSI+CiAgICAgICAgPHBhdGgKICAgICAgICAgICAgZD0iTTggMmMxIDAgMTEgMCAxMiAwczIgMSAyIDJjMCAxIDAgMTEgMCAxMnMwIDItMiAyQzIwIDE0IDIwIDQgMjAgNFMxMCA0IDYgNGMwLTIgMS0yIDItMnoiIC8+CiAgICAgICAgPHBhdGgKICAgICAgICAgICAgZD0iTTE4IDhjMC0xLTEtMi0yLTJTNSA2IDQgNnMtMiAxLTIgMmMwIDEgMCAxMSAwIDEyczEgMiAyIDJjMSAwIDExIDAgMTIgMHMyLTEgMi0yYzAtMSAwLTExIDAtMTJ6bS0yIDB2MTJINFY4eiIgLz4KICAgICAgICA8cGF0aCBkPSJNNiAxM3YyaDh2LTJ6IiAvPgogICAgPC9nPgo8L3N2Zz4K);
  --jp-icon-console: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIwMCAyMDAiPgogIDxnIGNsYXNzPSJqcC1jb25zb2xlLWljb24tYmFja2dyb3VuZC1jb2xvciBqcC1pY29uLXNlbGVjdGFibGUiIGZpbGw9IiMwMjg4RDEiPgogICAgPHBhdGggZD0iTTIwIDE5LjhoMTYwdjE1OS45SDIweiIvPgogIDwvZz4KICA8ZyBjbGFzcz0ianAtY29uc29sZS1pY29uLWNvbG9yIGpwLWljb24tc2VsZWN0YWJsZS1pbnZlcnNlIiBmaWxsPSIjZmZmIj4KICAgIDxwYXRoIGQ9Ik0xMDUgMTI3LjNoNDB2MTIuOGgtNDB6TTUxLjEgNzdMNzQgOTkuOWwtMjMuMyAyMy4zIDEwLjUgMTAuNSAyMy4zLTIzLjNMOTUgOTkuOSA4NC41IDg5LjQgNjEuNiA2Ni41eiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-copy: url(data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMTggMTgiIHdpZHRoPSIxNiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTExLjksMUgzLjJDMi40LDEsMS43LDEuNywxLjcsMi41djEwLjJoMS41VjIuNWg4LjdWMXogTTE0LjEsMy45aC04Yy0wLjgsMC0xLjUsMC43LTEuNSwxLjV2MTAuMmMwLDAuOCwwLjcsMS41LDEuNSwxLjVoOCBjMC44LDAsMS41LTAuNywxLjUtMS41VjUuNEMxNS41LDQuNiwxNC45LDMuOSwxNC4xLDMuOXogTTE0LjEsMTUuNWgtOFY1LjRoOFYxNS41eiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-copyright: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIGVuYWJsZS1iYWNrZ3JvdW5kPSJuZXcgMCAwIDI0IDI0IiBoZWlnaHQ9IjI0IiB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIyNCI+CiAgPGcgY2xhc3M9ImpwLWljb24zIiBmaWxsPSIjNjE2MTYxIj4KICAgIDxwYXRoIGQ9Ik0xMS44OCw5LjE0YzEuMjgsMC4wNiwxLjYxLDEuMTUsMS42MywxLjY2aDEuNzljLTAuMDgtMS45OC0xLjQ5LTMuMTktMy40NS0zLjE5QzkuNjQsNy42MSw4LDksOCwxMi4xNCBjMCwxLjk0LDAuOTMsNC4yNCwzLjg0LDQuMjRjMi4yMiwwLDMuNDEtMS42NSwzLjQ0LTIuOTVoLTEuNzljLTAuMDMsMC41OS0wLjQ1LDEuMzgtMS42MywxLjQ0QzEwLjU1LDE0LjgzLDEwLDEzLjgxLDEwLDEyLjE0IEMxMCw5LjI1LDExLjI4LDkuMTYsMTEuODgsOS4xNHogTTEyLDJDNi40OCwyLDIsNi40OCwyLDEyczQuNDgsMTAsMTAsMTBzMTAtNC40OCwxMC0xMFMxNy41MiwyLDEyLDJ6IE0xMiwyMGMtNC40MSwwLTgtMy41OS04LTggczMuNTktOCw4LThzOCwzLjU5LDgsOFMxNi40MSwyMCwxMiwyMHoiLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-cut: url(data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIxNiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTkuNjQgNy42NGMuMjMtLjUuMzYtMS4wNS4zNi0xLjY0IDAtMi4yMS0xLjc5LTQtNC00UzIgMy43OSAyIDZzMS43OSA0IDQgNGMuNTkgMCAxLjE0LS4xMyAxLjY0LS4zNkwxMCAxMmwtMi4zNiAyLjM2QzcuMTQgMTQuMTMgNi41OSAxNCA2IDE0Yy0yLjIxIDAtNCAxLjc5LTQgNHMxLjc5IDQgNCA0IDQtMS43OSA0LTRjMC0uNTktLjEzLTEuMTQtLjM2LTEuNjRMMTIgMTRsNyA3aDN2LTFMOS42NCA3LjY0ek02IDhjLTEuMSAwLTItLjg5LTItMnMuOS0yIDItMiAyIC44OSAyIDItLjkgMi0yIDJ6bTAgMTJjLTEuMSAwLTItLjg5LTItMnMuOS0yIDItMiAyIC44OSAyIDItLjkgMi0yIDJ6bTYtNy41Yy0uMjggMC0uNS0uMjItLjUtLjVzLjIyLS41LjUtLjUuNS4yMi41LjUtLjIyLjUtLjUuNXpNMTkgM2wtNiA2IDIgMiA3LTdWM3oiLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-delete: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgd2lkdGg9IjE2cHgiIGhlaWdodD0iMTZweCI+CiAgICA8cGF0aCBkPSJNMCAwaDI0djI0SDB6IiBmaWxsPSJub25lIiAvPgogICAgPHBhdGggY2xhc3M9ImpwLWljb24zIiBmaWxsPSIjNjI2MjYyIiBkPSJNNiAxOWMwIDEuMS45IDIgMiAyaDhjMS4xIDAgMi0uOSAyLTJWN0g2djEyek0xOSA0aC0zLjVsLTEtMWgtNWwtMSAxSDV2MmgxNFY0eiIgLz4KPC9zdmc+Cg==);
  --jp-icon-download: url(data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIxNiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTE5IDloLTRWM0g5djZINWw3IDcgNy03ek01IDE4djJoMTR2LTJINXoiLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-duplicate: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTQiIGhlaWdodD0iMTQiIHZpZXdCb3g9IjAgMCAxNCAxNCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggY2xhc3M9ImpwLWljb24zIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiIGNsaXAtcnVsZT0iZXZlbm9kZCIgZD0iTTIuNzk5OTggMC44NzVIOC44OTU4MkM5LjIwMDYxIDAuODc1IDkuNDQ5OTggMS4xMzkxNCA5LjQ0OTk4IDEuNDYxOThDOS40NDk5OCAxLjc4NDgyIDkuMjAwNjEgMi4wNDg5NiA4Ljg5NTgyIDIuMDQ4OTZIMy4zNTQxNUMzLjA0OTM2IDIuMDQ4OTYgMi43OTk5OCAyLjMxMzEgMi43OTk5OCAyLjYzNTk0VjkuNjc5NjlDMi43OTk5OCAxMC4wMDI1IDIuNTUwNjEgMTAuMjY2NyAyLjI0NTgyIDEwLjI2NjdDMS45NDEwMyAxMC4yNjY3IDEuNjkxNjUgMTAuMDAyNSAxLjY5MTY1IDkuNjc5NjlWMi4wNDg5NkMxLjY5MTY1IDEuNDAzMjggMi4xOTA0IDAuODc1IDIuNzk5OTggMC44NzVaTTUuMzY2NjUgMTEuOVY0LjU1SDExLjA4MzNWMTEuOUg1LjM2NjY1Wk00LjE0MTY1IDQuMTQxNjdDNC4xNDE2NSAzLjY5MDYzIDQuNTA3MjggMy4zMjUgNC45NTgzMiAzLjMyNUgxMS40OTE3QzExLjk0MjcgMy4zMjUgMTIuMzA4MyAzLjY5MDYzIDEyLjMwODMgNC4xNDE2N1YxMi4zMDgzQzEyLjMwODMgMTIuNzU5NCAxMS45NDI3IDEzLjEyNSAxMS40OTE3IDEzLjEyNUg0Ljk1ODMyQzQuNTA3MjggMTMuMTI1IDQuMTQxNjUgMTIuNzU5NCA0LjE0MTY1IDEyLjMwODNWNC4xNDE2N1oiIGZpbGw9IiM2MTYxNjEiLz4KPHBhdGggY2xhc3M9ImpwLWljb24zIiBkPSJNOS40MzU3NCA4LjI2NTA3SDguMzY0MzFWOS4zMzY1QzguMzY0MzEgOS40NTQzNSA4LjI2Nzg4IDkuNTUwNzggOC4xNTAwMiA5LjU1MDc4QzguMDMyMTcgOS41NTA3OCA3LjkzNTc0IDkuNDU0MzUgNy45MzU3NCA5LjMzNjVWOC4yNjUwN0g2Ljg2NDMxQzYuNzQ2NDUgOC4yNjUwNyA2LjY1MDAyIDguMTY4NjQgNi42NTAwMiA4LjA1MDc4QzYuNjUwMDIgNy45MzI5MiA2Ljc0NjQ1IDcuODM2NSA2Ljg2NDMxIDcuODM2NUg3LjkzNTc0VjYuNzY1MDdDNy45MzU3NCA2LjY0NzIxIDguMDMyMTcgNi41NTA3OCA4LjE1MDAyIDYuNTUwNzhDOC4yNjc4OCA2LjU1MDc4IDguMzY0MzEgNi42NDcyMSA4LjM2NDMxIDYuNzY1MDdWNy44MzY1SDkuNDM1NzRDOS41NTM2IDcuODM2NSA5LjY1MDAyIDcuOTMyOTIgOS42NTAwMiA4LjA1MDc4QzkuNjUwMDIgOC4xNjg2NCA5LjU1MzYgOC4yNjUwNyA5LjQzNTc0IDguMjY1MDdaIiBmaWxsPSIjNjE2MTYxIiBzdHJva2U9IiM2MTYxNjEiIHN0cm9rZS13aWR0aD0iMC41Ii8+Cjwvc3ZnPgo=);
  --jp-icon-edit: url(data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIxNiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTMgMTcuMjVWMjFoMy43NUwxNy44MSA5Ljk0bC0zLjc1LTMuNzVMMyAxNy4yNXpNMjAuNzEgNy4wNGMuMzktLjM5LjM5LTEuMDIgMC0xLjQxbC0yLjM0LTIuMzRjLS4zOS0uMzktMS4wMi0uMzktMS40MSAwbC0xLjgzIDEuODMgMy43NSAzLjc1IDEuODMtMS44M3oiLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-ellipses: url(data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIxNiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPGNpcmNsZSBjeD0iNSIgY3k9IjEyIiByPSIyIi8+CiAgICA8Y2lyY2xlIGN4PSIxMiIgY3k9IjEyIiByPSIyIi8+CiAgICA8Y2lyY2xlIGN4PSIxOSIgY3k9IjEyIiByPSIyIi8+CiAgPC9nPgo8L3N2Zz4K);
  --jp-icon-error: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KPGcgY2xhc3M9ImpwLWljb24zIiBmaWxsPSIjNjE2MTYxIj48Y2lyY2xlIGN4PSIxMiIgY3k9IjE5IiByPSIyIi8+PHBhdGggZD0iTTEwIDNoNHYxMmgtNHoiLz48L2c+CjxwYXRoIGZpbGw9Im5vbmUiIGQ9Ik0wIDBoMjR2MjRIMHoiLz4KPC9zdmc+Cg==);
  --jp-icon-expand-all: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICAgIDxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSI+CiAgICAgICAgPHBhdGgKICAgICAgICAgICAgZD0iTTggMmMxIDAgMTEgMCAxMiAwczIgMSAyIDJjMCAxIDAgMTEgMCAxMnMwIDItMiAyQzIwIDE0IDIwIDQgMjAgNFMxMCA0IDYgNGMwLTIgMS0yIDItMnoiIC8+CiAgICAgICAgPHBhdGgKICAgICAgICAgICAgZD0iTTE4IDhjMC0xLTEtMi0yLTJTNSA2IDQgNnMtMiAxLTIgMmMwIDEgMCAxMSAwIDEyczEgMiAyIDJjMSAwIDExIDAgMTIgMHMyLTEgMi0yYzAtMSAwLTExIDAtMTJ6bS0yIDB2MTJINFY4eiIgLz4KICAgICAgICA8cGF0aCBkPSJNMTEgMTBIOXYzSDZ2MmgzdjNoMnYtM2gzdi0yaC0zeiIgLz4KICAgIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-extension: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTIwLjUgMTFIMTlWN2MwLTEuMS0uOS0yLTItMmgtNFYzLjVDMTMgMi4xMiAxMS44OCAxIDEwLjUgMVM4IDIuMTIgOCAzLjVWNUg0Yy0xLjEgMC0xLjk5LjktMS45OSAydjMuOEgzLjVjMS40OSAwIDIuNyAxLjIxIDIuNyAyLjdzLTEuMjEgMi43LTIuNyAyLjdIMlYyMGMwIDEuMS45IDIgMiAyaDMuOHYtMS41YzAtMS40OSAxLjIxLTIuNyAyLjctMi43IDEuNDkgMCAyLjcgMS4yMSAyLjcgMi43VjIySDE3YzEuMSAwIDItLjkgMi0ydi00aDEuNWMxLjM4IDAgMi41LTEuMTIgMi41LTIuNVMyMS44OCAxMSAyMC41IDExeiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-fast-forward: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICAgIDxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSI+CiAgICAgICAgPHBhdGggZD0iTTQgMThsOC41LTZMNCA2djEyem05LTEydjEybDguNS02TDEzIDZ6Ii8+CiAgICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-file-upload: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTkgMTZoNnYtNmg0bC03LTctNyA3aDR6bS00IDJoMTR2Mkg1eiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-file: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIyIDIyIj4KICA8cGF0aCBjbGFzcz0ianAtaWNvbjMganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjNjE2MTYxIiBkPSJNMTkuMyA4LjJsLTUuNS01LjVjLS4zLS4zLS43LS41LTEuMi0uNUgzLjljLS44LjEtMS42LjktMS42IDEuOHYxNC4xYzAgLjkuNyAxLjYgMS42IDEuNmgxNC4yYy45IDAgMS42LS43IDEuNi0xLjZWOS40Yy4xLS41LS4xLS45LS40LTEuMnptLTUuOC0zLjNsMy40IDMuNmgtMy40VjQuOXptMy45IDEyLjdINC43Yy0uMSAwLS4yIDAtLjItLjJWNC43YzAtLjIuMS0uMy4yLS4zaDcuMnY0LjRzMCAuOC4zIDEuMWMuMy4zIDEuMS4zIDEuMS4zaDQuM3Y3LjJzLS4xLjItLjIuMnoiLz4KPC9zdmc+Cg==);
  --jp-icon-filter-dot: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiNGRkYiPgogICAgPHBhdGggZD0iTTE0LDEyVjE5Ljg4QzE0LjA0LDIwLjE4IDEzLjk0LDIwLjUgMTMuNzEsMjAuNzFDMTMuMzIsMjEuMSAxMi42OSwyMS4xIDEyLjMsMjAuNzFMMTAuMjksMTguN0MxMC4wNiwxOC40NyA5Ljk2LDE4LjE2IDEwLDE3Ljg3VjEySDkuOTdMNC4yMSw0LjYyQzMuODcsNC4xOSAzLjk1LDMuNTYgNC4zOCwzLjIyQzQuNTcsMy4wOCA0Ljc4LDMgNSwzVjNIMTlWM0MxOS4yMiwzIDE5LjQzLDMuMDggMTkuNjIsMy4yMkMyMC4wNSwzLjU2IDIwLjEzLDQuMTkgMTkuNzksNC42MkwxNC4wMywxMkgxNFoiIC8+CiAgPC9nPgogIDxnIGNsYXNzPSJqcC1pY29uLWRvdCIgZmlsbD0iI0ZGRiI+CiAgICA8Y2lyY2xlIGN4PSIxOCIgY3k9IjE3IiByPSIzIj48L2NpcmNsZT4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-filter-list: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTEwIDE4aDR2LTJoLTR2MnpNMyA2djJoMThWNkgzem0zIDdoMTJ2LTJINnYyeiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-filter: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiNGRkYiPgogICAgPHBhdGggZD0iTTE0LDEyVjE5Ljg4QzE0LjA0LDIwLjE4IDEzLjk0LDIwLjUgMTMuNzEsMjAuNzFDMTMuMzIsMjEuMSAxMi42OSwyMS4xIDEyLjMsMjAuNzFMMTAuMjksMTguN0MxMC4wNiwxOC40NyA5Ljk2LDE4LjE2IDEwLDE3Ljg3VjEySDkuOTdMNC4yMSw0LjYyQzMuODcsNC4xOSAzLjk1LDMuNTYgNC4zOCwzLjIyQzQuNTcsMy4wOCA0Ljc4LDMgNSwzVjNIMTlWM0MxOS4yMiwzIDE5LjQzLDMuMDggMTkuNjIsMy4yMkMyMC4wNSwzLjU2IDIwLjEzLDQuMTkgMTkuNzksNC42MkwxNC4wMywxMkgxNFoiIC8+CiAgPC9nPgo8L3N2Zz4K);
  --jp-icon-folder-favorite: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIGhlaWdodD0iMjRweCIgdmlld0JveD0iMCAwIDI0IDI0IiB3aWR0aD0iMjRweCIgZmlsbD0iIzAwMDAwMCI+CiAgPHBhdGggZD0iTTAgMGgyNHYyNEgwVjB6IiBmaWxsPSJub25lIi8+PHBhdGggY2xhc3M9ImpwLWljb24zIGpwLWljb24tc2VsZWN0YWJsZSIgZmlsbD0iIzYxNjE2MSIgZD0iTTIwIDZoLThsLTItMkg0Yy0xLjEgMC0yIC45LTIgMnYxMmMwIDEuMS45IDIgMiAyaDE2YzEuMSAwIDItLjkgMi0yVjhjMC0xLjEtLjktMi0yLTJ6bS0yLjA2IDExTDE1IDE1LjI4IDEyLjA2IDE3bC43OC0zLjMzLTIuNTktMi4yNCAzLjQxLS4yOUwxNSA4bDEuMzQgMy4xNCAzLjQxLjI5LTIuNTkgMi4yNC43OCAzLjMzeiIvPgo8L3N2Zz4K);
  --jp-icon-folder: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8cGF0aCBjbGFzcz0ianAtaWNvbjMganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjNjE2MTYxIiBkPSJNMTAgNEg0Yy0xLjEgMC0xLjk5LjktMS45OSAyTDIgMThjMCAxLjEuOSAyIDIgMmgxNmMxLjEgMCAyLS45IDItMlY4YzAtMS4xLS45LTItMi0yaC04bC0yLTJ6Ii8+Cjwvc3ZnPgo=);
  --jp-icon-home: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIGhlaWdodD0iMjRweCIgdmlld0JveD0iMCAwIDI0IDI0IiB3aWR0aD0iMjRweCIgZmlsbD0iIzAwMDAwMCI+CiAgPHBhdGggZD0iTTAgMGgyNHYyNEgweiIgZmlsbD0ibm9uZSIvPjxwYXRoIGNsYXNzPSJqcC1pY29uMyBqcC1pY29uLXNlbGVjdGFibGUiIGZpbGw9IiM2MTYxNjEiIGQ9Ik0xMCAyMHYtNmg0djZoNXYtOGgzTDEyIDMgMiAxMmgzdjh6Ii8+Cjwvc3ZnPgo=);
  --jp-icon-html5: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDUxMiA1MTIiPgogIDxwYXRoIGNsYXNzPSJqcC1pY29uMCBqcC1pY29uLXNlbGVjdGFibGUiIGZpbGw9IiMwMDAiIGQ9Ik0xMDguNCAwaDIzdjIyLjhoMjEuMlYwaDIzdjY5aC0yM1Y0NmgtMjF2MjNoLTIzLjJNMjA2IDIzaC0yMC4zVjBoNjMuN3YyM0gyMjl2NDZoLTIzbTUzLjUtNjloMjQuMWwxNC44IDI0LjNMMzEzLjIgMGgyNC4xdjY5aC0yM1YzNC44bC0xNi4xIDI0LjgtMTYuMS0yNC44VjY5aC0yMi42bTg5LjItNjloMjN2NDYuMmgzMi42VjY5aC01NS42Ii8+CiAgPHBhdGggY2xhc3M9ImpwLWljb24tc2VsZWN0YWJsZSIgZmlsbD0iI2U0NGQyNiIgZD0iTTEwNy42IDQ3MWwtMzMtMzcwLjRoMzYyLjhsLTMzIDM3MC4yTDI1NS43IDUxMiIvPgogIDxwYXRoIGNsYXNzPSJqcC1pY29uLXNlbGVjdGFibGUiIGZpbGw9IiNmMTY1MjkiIGQ9Ik0yNTYgNDgwLjVWMTMxaDE0OC4zTDM3NiA0NDciLz4KICA8cGF0aCBjbGFzcz0ianAtaWNvbi1zZWxlY3RhYmxlLWludmVyc2UiIGZpbGw9IiNlYmViZWIiIGQ9Ik0xNDIgMTc2LjNoMTE0djQ1LjRoLTY0LjJsNC4yIDQ2LjVoNjB2NDUuM0gxNTQuNG0yIDIyLjhIMjAybDMuMiAzNi4zIDUwLjggMTMuNnY0Ny40bC05My4yLTI2Ii8+CiAgPHBhdGggY2xhc3M9ImpwLWljb24tc2VsZWN0YWJsZS1pbnZlcnNlIiBmaWxsPSIjZmZmIiBkPSJNMzY5LjYgMTc2LjNIMjU1Ljh2NDUuNGgxMDkuNm0tNC4xIDQ2LjVIMjU1Ljh2NDUuNGg1NmwtNS4zIDU5LTUwLjcgMTMuNnY0Ny4ybDkzLTI1LjgiLz4KPC9zdmc+Cg==);
  --jp-icon-image: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIyIDIyIj4KICA8cGF0aCBjbGFzcz0ianAtaWNvbi1icmFuZDQganAtaWNvbi1zZWxlY3RhYmxlLWludmVyc2UiIGZpbGw9IiNGRkYiIGQ9Ik0yLjIgMi4yaDE3LjV2MTcuNUgyLjJ6Ii8+CiAgPHBhdGggY2xhc3M9ImpwLWljb24tYnJhbmQwIGpwLWljb24tc2VsZWN0YWJsZSIgZmlsbD0iIzNGNTFCNSIgZD0iTTIuMiAyLjJ2MTcuNWgxNy41bC4xLTE3LjVIMi4yem0xMi4xIDIuMmMxLjIgMCAyLjIgMSAyLjIgMi4ycy0xIDIuMi0yLjIgMi4yLTIuMi0xLTIuMi0yLjIgMS0yLjIgMi4yLTIuMnpNNC40IDE3LjZsMy4zLTguOCAzLjMgNi42IDIuMi0zLjIgNC40IDUuNEg0LjR6Ii8+Cjwvc3ZnPgo=);
  --jp-icon-info: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDUwLjk3OCA1MC45NzgiPgoJPGcgY2xhc3M9ImpwLWljb24zIiBmaWxsPSIjNjE2MTYxIj4KCQk8cGF0aCBkPSJNNDMuNTIsNy40NThDMzguNzExLDIuNjQ4LDMyLjMwNywwLDI1LjQ4OSwwQzE4LjY3LDAsMTIuMjY2LDIuNjQ4LDcuNDU4LDcuNDU4CgkJCWMtOS45NDMsOS45NDEtOS45NDMsMjYuMTE5LDAsMzYuMDYyYzQuODA5LDQuODA5LDExLjIxMiw3LjQ1NiwxOC4wMzEsNy40NThjMCwwLDAuMDAxLDAsMC4wMDIsMAoJCQljNi44MTYsMCwxMy4yMjEtMi42NDgsMTguMDI5LTcuNDU4YzQuODA5LTQuODA5LDcuNDU3LTExLjIxMiw3LjQ1Ny0xOC4wM0M1MC45NzcsMTguNjcsNDguMzI4LDEyLjI2Niw0My41Miw3LjQ1OHoKCQkJIE00Mi4xMDYsNDIuMTA1Yy00LjQzMiw0LjQzMS0xMC4zMzIsNi44NzItMTYuNjE1LDYuODcyaC0wLjAwMmMtNi4yODUtMC4wMDEtMTIuMTg3LTIuNDQxLTE2LjYxNy02Ljg3MgoJCQljLTkuMTYyLTkuMTYzLTkuMTYyLTI0LjA3MSwwLTMzLjIzM0MxMy4zMDMsNC40NCwxOS4yMDQsMiwyNS40ODksMmM2LjI4NCwwLDEyLjE4NiwyLjQ0LDE2LjYxNyw2Ljg3MgoJCQljNC40MzEsNC40MzEsNi44NzEsMTAuMzMyLDYuODcxLDE2LjYxN0M0OC45NzcsMzEuNzcyLDQ2LjUzNiwzNy42NzUsNDIuMTA2LDQyLjEwNXoiLz4KCQk8cGF0aCBkPSJNMjMuNTc4LDMyLjIxOGMtMC4wMjMtMS43MzQsMC4xNDMtMy4wNTksMC40OTYtMy45NzJjMC4zNTMtMC45MTMsMS4xMS0xLjk5NywyLjI3Mi0zLjI1MwoJCQljMC40NjgtMC41MzYsMC45MjMtMS4wNjIsMS4zNjctMS41NzVjMC42MjYtMC43NTMsMS4xMDQtMS40NzgsMS40MzYtMi4xNzVjMC4zMzEtMC43MDcsMC40OTUtMS41NDEsMC40OTUtMi41CgkJCWMwLTEuMDk2LTAuMjYtMi4wODgtMC43NzktMi45NzljLTAuNTY1LTAuODc5LTEuNTAxLTEuMzM2LTIuODA2LTEuMzY5Yy0xLjgwMiwwLjA1Ny0yLjk4NSwwLjY2Ny0zLjU1LDEuODMyCgkJCWMtMC4zMDEsMC41MzUtMC41MDMsMS4xNDEtMC42MDcsMS44MTRjLTAuMTM5LDAuNzA3LTAuMjA3LDEuNDMyLTAuMjA3LDIuMTc0aC0yLjkzN2MtMC4wOTEtMi4yMDgsMC40MDctNC4xMTQsMS40OTMtNS43MTkKCQkJYzEuMDYyLTEuNjQsMi44NTUtMi40ODEsNS4zNzgtMi41MjdjMi4xNiwwLjAyMywzLjg3NCwwLjYwOCw1LjE0MSwxLjc1OGMxLjI3OCwxLjE2LDEuOTI5LDIuNzY0LDEuOTUsNC44MTEKCQkJYzAsMS4xNDItMC4xMzcsMi4xMTEtMC40MSwyLjkxMWMtMC4zMDksMC44NDUtMC43MzEsMS41OTMtMS4yNjgsMi4yNDNjLTAuNDkyLDAuNjUtMS4wNjgsMS4zMTgtMS43MywyLjAwMgoJCQljLTAuNjUsMC42OTctMS4zMTMsMS40NzktMS45ODcsMi4zNDZjLTAuMjM5LDAuMzc3LTAuNDI5LDAuNzc3LTAuNTY1LDEuMTk5Yy0wLjE2LDAuOTU5LTAuMjE3LDEuOTUxLTAuMTcxLDIuOTc5CgkJCUMyNi41ODksMzIuMjE4LDIzLjU3OCwzMi4yMTgsMjMuNTc4LDMyLjIxOHogTTIzLjU3OCwzOC4yMnYtMy40ODRoMy4wNzZ2My40ODRIMjMuNTc4eiIvPgoJPC9nPgo8L3N2Zz4K);
  --jp-icon-inspector: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8cGF0aCBjbGFzcz0ianAtaW5zcGVjdG9yLWljb24tY29sb3IganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjNjE2MTYxIiBkPSJNMjAgNEg0Yy0xLjEgMC0xLjk5LjktMS45OSAyTDIgMThjMCAxLjEuOSAyIDIgMmgxNmMxLjEgMCAyLS45IDItMlY2YzAtMS4xLS45LTItMi0yem0tNSAxNEg0di00aDExdjR6bTAtNUg0VjloMTF2NHptNSA1aC00VjloNHY5eiIvPgo8L3N2Zz4K);
  --jp-icon-json: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIyIDIyIj4KICA8ZyBjbGFzcz0ianAtanNvbi1pY29uLWNvbG9yIGpwLWljb24tc2VsZWN0YWJsZSIgZmlsbD0iI0Y5QTgyNSI+CiAgICA8cGF0aCBkPSJNMjAuMiAxMS44Yy0xLjYgMC0xLjcuNS0xLjcgMSAwIC40LjEuOS4xIDEuMy4xLjUuMS45LjEgMS4zIDAgMS43LTEuNCAyLjMtMy41IDIuM2gtLjl2LTEuOWguNWMxLjEgMCAxLjQgMCAxLjQtLjggMC0uMyAwLS42LS4xLTEgMC0uNC0uMS0uOC0uMS0xLjIgMC0xLjMgMC0xLjggMS4zLTItMS4zLS4yLTEuMy0uNy0xLjMtMiAwLS40LjEtLjguMS0xLjIuMS0uNC4xLS43LjEtMSAwLS44LS40LS43LTEuNC0uOGgtLjVWNC4xaC45YzIuMiAwIDMuNS43IDMuNSAyLjMgMCAuNC0uMS45LS4xIDEuMy0uMS41LS4xLjktLjEgMS4zIDAgLjUuMiAxIDEuNyAxdjEuOHpNMS44IDEwLjFjMS42IDAgMS43LS41IDEuNy0xIDAtLjQtLjEtLjktLjEtMS4zLS4xLS41LS4xLS45LS4xLTEuMyAwLTEuNiAxLjQtMi4zIDMuNS0yLjNoLjl2MS45aC0uNWMtMSAwLTEuNCAwLTEuNC44IDAgLjMgMCAuNi4xIDEgMCAuMi4xLjYuMSAxIDAgMS4zIDAgMS44LTEuMyAyQzYgMTEuMiA2IDExLjcgNiAxM2MwIC40LS4xLjgtLjEgMS4yLS4xLjMtLjEuNy0uMSAxIDAgLjguMy44IDEuNC44aC41djEuOWgtLjljLTIuMSAwLTMuNS0uNi0zLjUtMi4zIDAtLjQuMS0uOS4xLTEuMy4xLS41LjEtLjkuMS0xLjMgMC0uNS0uMi0xLTEuNy0xdi0xLjl6Ii8+CiAgICA8Y2lyY2xlIGN4PSIxMSIgY3k9IjEzLjgiIHI9IjIuMSIvPgogICAgPGNpcmNsZSBjeD0iMTEiIGN5PSI4LjIiIHI9IjIuMSIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-julia: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDMyNSAzMDAiPgogIDxnIGNsYXNzPSJqcC1icmFuZDAganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjY2IzYzMzIj4KICAgIDxwYXRoIGQ9Ik0gMTUwLjg5ODQzOCAyMjUgQyAxNTAuODk4NDM4IDI2Ni40MjE4NzUgMTE3LjMyMDMxMiAzMDAgNzUuODk4NDM4IDMwMCBDIDM0LjQ3NjU2MiAzMDAgMC44OTg0MzggMjY2LjQyMTg3NSAwLjg5ODQzOCAyMjUgQyAwLjg5ODQzOCAxODMuNTc4MTI1IDM0LjQ3NjU2MiAxNTAgNzUuODk4NDM4IDE1MCBDIDExNy4zMjAzMTIgMTUwIDE1MC44OTg0MzggMTgzLjU3ODEyNSAxNTAuODk4NDM4IDIyNSIvPgogIDwvZz4KICA8ZyBjbGFzcz0ianAtYnJhbmQwIGpwLWljb24tc2VsZWN0YWJsZSIgZmlsbD0iIzM4OTgyNiI+CiAgICA8cGF0aCBkPSJNIDIzNy41IDc1IEMgMjM3LjUgMTE2LjQyMTg3NSAyMDMuOTIxODc1IDE1MCAxNjIuNSAxNTAgQyAxMjEuMDc4MTI1IDE1MCA4Ny41IDExNi40MjE4NzUgODcuNSA3NSBDIDg3LjUgMzMuNTc4MTI1IDEyMS4wNzgxMjUgMCAxNjIuNSAwIEMgMjAzLjkyMTg3NSAwIDIzNy41IDMzLjU3ODEyNSAyMzcuNSA3NSIvPgogIDwvZz4KICA8ZyBjbGFzcz0ianAtYnJhbmQwIGpwLWljb24tc2VsZWN0YWJsZSIgZmlsbD0iIzk1NThiMiI+CiAgICA8cGF0aCBkPSJNIDMyNC4xMDE1NjIgMjI1IEMgMzI0LjEwMTU2MiAyNjYuNDIxODc1IDI5MC41MjM0MzggMzAwIDI0OS4xMDE1NjIgMzAwIEMgMjA3LjY3OTY4OCAzMDAgMTc0LjEwMTU2MiAyNjYuNDIxODc1IDE3NC4xMDE1NjIgMjI1IEMgMTc0LjEwMTU2MiAxODMuNTc4MTI1IDIwNy42Nzk2ODggMTUwIDI0OS4xMDE1NjIgMTUwIEMgMjkwLjUyMzQzOCAxNTAgMzI0LjEwMTU2MiAxODMuNTc4MTI1IDMyNC4xMDE1NjIgMjI1Ii8+CiAgPC9nPgo8L3N2Zz4K);
  --jp-icon-jupyter-favicon: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTUyIiBoZWlnaHQ9IjE2NSIgdmlld0JveD0iMCAwIDE1MiAxNjUiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICAgPGcgY2xhc3M9ImpwLWp1cHl0ZXItaWNvbi1jb2xvciIgZmlsbD0iI0YzNzcyNiI+CiAgICA8cGF0aCB0cmFuc2Zvcm09InRyYW5zbGF0ZSgwLjA3ODk0NywgMTEwLjU4MjkyNykiIGQ9Ik03NS45NDIyODQyLDI5LjU4MDQ1NjEgQzQzLjMwMjM5NDcsMjkuNTgwNDU2MSAxNC43OTY3ODMyLDE3LjY1MzQ2MzQgMCwwIEM1LjUxMDgzMjExLDE1Ljg0MDY4MjkgMTUuNzgxNTM4OSwyOS41NjY3NzMyIDI5LjM5MDQ5NDcsMzkuMjc4NDE3MSBDNDIuOTk5Nyw0OC45ODk4NTM3IDU5LjI3MzcsNTQuMjA2NzgwNSA3NS45NjA1Nzg5LDU0LjIwNjc4MDUgQzkyLjY0NzQ1NzksNTQuMjA2NzgwNSAxMDguOTIxNDU4LDQ4Ljk4OTg1MzcgMTIyLjUzMDY2MywzOS4yNzg0MTcxIEMxMzYuMTM5NDUzLDI5LjU2Njc3MzIgMTQ2LjQxMDI4NCwxNS44NDA2ODI5IDE1MS45MjExNTgsMCBDMTM3LjA4Nzg2OCwxNy42NTM0NjM0IDEwOC41ODI1ODksMjkuNTgwNDU2MSA3NS45NDIyODQyLDI5LjU4MDQ1NjEgTDc1Ljk0MjI4NDIsMjkuNTgwNDU2MSBaIiAvPgogICAgPHBhdGggdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMC4wMzczNjgsIDAuNzA0ODc4KSIgZD0iTTc1Ljk3ODQ1NzksMjQuNjI2NDA3MyBDMTA4LjYxODc2MywyNC42MjY0MDczIDEzNy4xMjQ0NTgsMzYuNTUzNDQxNSAxNTEuOTIxMTU4LDU0LjIwNjc4MDUgQzE0Ni40MTAyODQsMzguMzY2MjIyIDEzNi4xMzk0NTMsMjQuNjQwMTMxNyAxMjIuNTMwNjYzLDE0LjkyODQ4NzggQzEwOC45MjE0NTgsNS4yMTY4NDM5IDkyLjY0NzQ1NzksMCA3NS45NjA1Nzg5LDAgQzU5LjI3MzcsMCA0Mi45OTk3LDUuMjE2ODQzOSAyOS4zOTA0OTQ3LDE0LjkyODQ4NzggQzE1Ljc4MTUzODksMjQuNjQwMTMxNyA1LjUxMDgzMjExLDM4LjM2NjIyMiAwLDU0LjIwNjc4MDUgQzE0LjgzMzA4MTYsMzYuNTg5OTI5MyA0My4zMzg1Njg0LDI0LjYyNjQwNzMgNzUuOTc4NDU3OSwyNC42MjY0MDczIEw3NS45Nzg0NTc5LDI0LjYyNjQwNzMgWiIgLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-jupyter: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzkiIGhlaWdodD0iNTEiIHZpZXdCb3g9IjAgMCAzOSA1MSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyB0cmFuc2Zvcm09InRyYW5zbGF0ZSgtMTYzOCAtMjI4MSkiPgogICAgIDxnIGNsYXNzPSJqcC1qdXB5dGVyLWljb24tY29sb3IiIGZpbGw9IiNGMzc3MjYiPgogICAgICA8cGF0aCB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxNjM5Ljc0IDIzMTEuOTgpIiBkPSJNIDE4LjI2NDYgNy4xMzQxMUMgMTAuNDE0NSA3LjEzNDExIDMuNTU4NzIgNC4yNTc2IDAgMEMgMS4zMjUzOSAzLjgyMDQgMy43OTU1NiA3LjEzMDgxIDcuMDY4NiA5LjQ3MzAzQyAxMC4zNDE3IDExLjgxNTIgMTQuMjU1NyAxMy4wNzM0IDE4LjI2OSAxMy4wNzM0QyAyMi4yODIzIDEzLjA3MzQgMjYuMTk2MyAxMS44MTUyIDI5LjQ2OTQgOS40NzMwM0MgMzIuNzQyNCA3LjEzMDgxIDM1LjIxMjYgMy44MjA0IDM2LjUzOCAwQyAzMi45NzA1IDQuMjU3NiAyNi4xMTQ4IDcuMTM0MTEgMTguMjY0NiA3LjEzNDExWiIvPgogICAgICA8cGF0aCB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxNjM5LjczIDIyODUuNDgpIiBkPSJNIDE4LjI3MzMgNS45MzkzMUMgMjYuMTIzNSA1LjkzOTMxIDMyLjk3OTMgOC44MTU4MyAzNi41MzggMTMuMDczNEMgMzUuMjEyNiA5LjI1MzAzIDMyLjc0MjQgNS45NDI2MiAyOS40Njk0IDMuNjAwNEMgMjYuMTk2MyAxLjI1ODE4IDIyLjI4MjMgMCAxOC4yNjkgMEMgMTQuMjU1NyAwIDEwLjM0MTcgMS4yNTgxOCA3LjA2ODYgMy42MDA0QyAzLjc5NTU2IDUuOTQyNjIgMS4zMjUzOSA5LjI1MzAzIDAgMTMuMDczNEMgMy41Njc0NSA4LjgyNDYzIDEwLjQyMzIgNS45MzkzMSAxOC4yNzMzIDUuOTM5MzFaIi8+CiAgICA8L2c+CiAgICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgICA8cGF0aCB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxNjY5LjMgMjI4MS4zMSkiIGQ9Ik0gNS44OTM1MyAyLjg0NEMgNS45MTg4OSAzLjQzMTY1IDUuNzcwODUgNC4wMTM2NyA1LjQ2ODE1IDQuNTE2NDVDIDUuMTY1NDUgNS4wMTkyMiA0LjcyMTY4IDUuNDIwMTUgNC4xOTI5OSA1LjY2ODUxQyAzLjY2NDMgNS45MTY4OCAzLjA3NDQ0IDYuMDAxNTEgMi40OTgwNSA1LjkxMTcxQyAxLjkyMTY2IDUuODIxOSAxLjM4NDYzIDUuNTYxNyAwLjk1NDg5OCA1LjE2NDAxQyAwLjUyNTE3IDQuNzY2MzMgMC4yMjIwNTYgNC4yNDkwMyAwLjA4MzkwMzcgMy42Nzc1N0MgLTAuMDU0MjQ4MyAzLjEwNjExIC0wLjAyMTIzIDIuNTA2MTcgMC4xNzg3ODEgMS45NTM2NEMgMC4zNzg3OTMgMS40MDExIDAuNzM2ODA5IDAuOTIwODE3IDEuMjA3NTQgMC41NzM1MzhDIDEuNjc4MjYgMC4yMjYyNTkgMi4yNDA1NSAwLjAyNzU5MTkgMi44MjMyNiAwLjAwMjY3MjI5QyAzLjYwMzg5IC0wLjAzMDcxMTUgNC4zNjU3MyAwLjI0OTc4OSA0Ljk0MTQyIDAuNzgyNTUxQyA1LjUxNzExIDEuMzE1MzEgNS44NTk1NiAyLjA1Njc2IDUuODkzNTMgMi44NDRaIi8+CiAgICAgIDxwYXRoIHRyYW5zZm9ybT0idHJhbnNsYXRlKDE2MzkuOCAyMzIzLjgxKSIgZD0iTSA3LjQyNzg5IDMuNTgzMzhDIDcuNDYwMDggNC4zMjQzIDcuMjczNTUgNS4wNTgxOSA2Ljg5MTkzIDUuNjkyMTNDIDYuNTEwMzEgNi4zMjYwNyA1Ljk1MDc1IDYuODMxNTYgNS4yODQxMSA3LjE0NDZDIDQuNjE3NDcgNy40NTc2MyAzLjg3MzcxIDcuNTY0MTQgMy4xNDcwMiA3LjQ1MDYzQyAyLjQyMDMyIDcuMzM3MTIgMS43NDMzNiA3LjAwODcgMS4yMDE4NCA2LjUwNjk1QyAwLjY2MDMyOCA2LjAwNTIgMC4yNzg2MSA1LjM1MjY4IDAuMTA1MDE3IDQuNjMyMDJDIC0wLjA2ODU3NTcgMy45MTEzNSAtMC4wMjYyMzYxIDMuMTU0OTQgMC4yMjY2NzUgMi40NTg1NkMgMC40Nzk1ODcgMS43NjIxNyAwLjkzMTY5NyAxLjE1NzEzIDEuNTI1NzYgMC43MjAwMzNDIDIuMTE5ODMgMC4yODI5MzUgMi44MjkxNCAwLjAzMzQzOTUgMy41NjM4OSAwLjAwMzEzMzQ0QyA0LjU0NjY3IC0wLjAzNzQwMzMgNS41MDUyOSAwLjMxNjcwNiA2LjIyOTYxIDAuOTg3ODM1QyA2Ljk1MzkzIDEuNjU4OTYgNy4zODQ4NCAyLjU5MjM1IDcuNDI3ODkgMy41ODMzOEwgNy40Mjc4OSAzLjU4MzM4WiIvPgogICAgICA8cGF0aCB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxNjM4LjM2IDIyODYuMDYpIiBkPSJNIDIuMjc0NzEgNC4zOTYyOUMgMS44NDM2MyA0LjQxNTA4IDEuNDE2NzEgNC4zMDQ0NSAxLjA0Nzk5IDQuMDc4NDNDIDAuNjc5MjY4IDMuODUyNCAwLjM4NTMyOCAzLjUyMTE0IDAuMjAzMzcxIDMuMTI2NTZDIDAuMDIxNDEzNiAyLjczMTk4IC0wLjA0MDM3OTggMi4yOTE4MyAwLjAyNTgxMTYgMS44NjE4MUMgMC4wOTIwMDMxIDEuNDMxOCAwLjI4MzIwNCAxLjAzMTI2IDAuNTc1MjEzIDAuNzEwODgzQyAwLjg2NzIyMiAwLjM5MDUxIDEuMjQ2OTEgMC4xNjQ3MDggMS42NjYyMiAwLjA2MjA1OTJDIDIuMDg1NTMgLTAuMDQwNTg5NyAyLjUyNTYxIC0wLjAxNTQ3MTQgMi45MzA3NiAwLjEzNDIzNUMgMy4zMzU5MSAwLjI4Mzk0MSAzLjY4NzkyIDAuNTUxNTA1IDMuOTQyMjIgMC45MDMwNkMgNC4xOTY1MiAxLjI1NDYyIDQuMzQxNjkgMS42NzQzNiA0LjM1OTM1IDIuMTA5MTZDIDQuMzgyOTkgMi42OTEwNyA0LjE3Njc4IDMuMjU4NjkgMy43ODU5NyAzLjY4NzQ2QyAzLjM5NTE2IDQuMTE2MjQgMi44NTE2NiA0LjM3MTE2IDIuMjc0NzEgNC4zOTYyOUwgMi4yNzQ3MSA0LjM5NjI5WiIvPgogICAgPC9nPgogIDwvZz4+Cjwvc3ZnPgo=);
  --jp-icon-jupyterlab-wordmark: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyMDAiIHZpZXdCb3g9IjAgMCAxODYwLjggNDc1Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjIiIGZpbGw9IiM0RTRFNEUiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDQ4MC4xMzY0MDEsIDY0LjI3MTQ5MykiPgogICAgPGcgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMC4wMDAwMDAsIDU4Ljg3NTU2NikiPgogICAgICA8ZyB0cmFuc2Zvcm09InRyYW5zbGF0ZSgwLjA4NzYwMywgMC4xNDAyOTQpIj4KICAgICAgICA8cGF0aCBkPSJNLTQyNi45LDE2OS44YzAsNDguNy0zLjcsNjQuNy0xMy42LDc2LjRjLTEwLjgsMTAtMjUsMTUuNS0zOS43LDE1LjVsMy43LDI5IGMyMi44LDAuMyw0NC44LTcuOSw2MS45LTIzLjFjMTcuOC0xOC41LDI0LTQ0LjEsMjQtODMuM1YwSC00Mjd2MTcwLjFMLTQyNi45LDE2OS44TC00MjYuOSwxNjkuOHoiLz4KICAgICAgPC9nPgogICAgPC9nPgogICAgPGcgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMTU1LjA0NTI5NiwgNTYuODM3MTA0KSI+CiAgICAgIDxnIHRyYW5zZm9ybT0idHJhbnNsYXRlKDEuNTYyNDUzLCAxLjc5OTg0MikiPgogICAgICAgIDxwYXRoIGQ9Ik0tMzEyLDE0OGMwLDIxLDAsMzkuNSwxLjcsNTUuNGgtMzEuOGwtMi4xLTMzLjNoLTAuOGMtNi43LDExLjYtMTYuNCwyMS4zLTI4LDI3LjkgYy0xMS42LDYuNi0yNC44LDEwLTM4LjIsOS44Yy0zMS40LDAtNjktMTcuNy02OS04OVYwaDM2LjR2MTEyLjdjMCwzOC43LDExLjYsNjQuNyw0NC42LDY0LjdjMTAuMy0wLjIsMjAuNC0zLjUsMjguOS05LjQgYzguNS01LjksMTUuMS0xNC4zLDE4LjktMjMuOWMyLjItNi4xLDMuMy0xMi41LDMuMy0xOC45VjAuMmgzNi40VjE0OEgtMzEyTC0zMTIsMTQ4eiIvPgogICAgICA8L2c+CiAgICA8L2c+CiAgICA8ZyB0cmFuc2Zvcm09InRyYW5zbGF0ZSgzOTAuMDEzMzIyLCA1My40Nzk2MzgpIj4KICAgICAgPGcgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMS43MDY0NTgsIDAuMjMxNDI1KSI+CiAgICAgICAgPHBhdGggZD0iTS00NzguNiw3MS40YzAtMjYtMC44LTQ3LTEuNy02Ni43aDMyLjdsMS43LDM0LjhoMC44YzcuMS0xMi41LDE3LjUtMjIuOCwzMC4xLTI5LjcgYzEyLjUtNywyNi43LTEwLjMsNDEtOS44YzQ4LjMsMCw4NC43LDQxLjcsODQuNywxMDMuM2MwLDczLjEtNDMuNywxMDkuMi05MSwxMDkuMmMtMTIuMSwwLjUtMjQuMi0yLjItMzUtNy44IGMtMTAuOC01LjYtMTkuOS0xMy45LTI2LjYtMjQuMmgtMC44VjI5MWgtMzZ2LTIyMEwtNDc4LjYsNzEuNEwtNDc4LjYsNzEuNHogTS00NDIuNiwxMjUuNmMwLjEsNS4xLDAuNiwxMC4xLDEuNywxNS4xIGMzLDEyLjMsOS45LDIzLjMsMTkuOCwzMS4xYzkuOSw3LjgsMjIuMSwxMi4xLDM0LjcsMTIuMWMzOC41LDAsNjAuNy0zMS45LDYwLjctNzguNWMwLTQwLjctMjEuMS03NS42LTU5LjUtNzUuNiBjLTEyLjksMC40LTI1LjMsNS4xLTM1LjMsMTMuNGMtOS45LDguMy0xNi45LDE5LjctMTkuNiwzMi40Yy0xLjUsNC45LTIuMywxMC0yLjUsMTUuMVYxMjUuNkwtNDQyLjYsMTI1LjZMLTQ0Mi42LDEyNS42eiIvPgogICAgICA8L2c+CiAgICA8L2c+CiAgICA8ZyB0cmFuc2Zvcm09InRyYW5zbGF0ZSg2MDYuNzQwNzI2LCA1Ni44MzcxMDQpIj4KICAgICAgPGcgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMC43NTEyMjYsIDEuOTg5Mjk5KSI+CiAgICAgICAgPHBhdGggZD0iTS00NDAuOCwwbDQzLjcsMTIwLjFjNC41LDEzLjQsOS41LDI5LjQsMTIuOCw0MS43aDAuOGMzLjctMTIuMiw3LjktMjcuNywxMi44LTQyLjQgbDM5LjctMTE5LjJoMzguNUwtMzQ2LjksMTQ1Yy0yNiw2OS43LTQzLjcsMTA1LjQtNjguNiwxMjcuMmMtMTIuNSwxMS43LTI3LjksMjAtNDQuNiwyMy45bC05LjEtMzEuMSBjMTEuNy0zLjksMjIuNS0xMC4xLDMxLjgtMTguMWMxMy4yLTExLjEsMjMuNy0yNS4yLDMwLjYtNDEuMmMxLjUtMi44LDIuNS01LjcsMi45LTguOGMtMC4zLTMuMy0xLjItNi42LTIuNS05LjdMLTQ4MC4yLDAuMSBoMzkuN0wtNDQwLjgsMEwtNDQwLjgsMHoiLz4KICAgICAgPC9nPgogICAgPC9nPgogICAgPGcgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoODIyLjc0ODEwNCwgMC4wMDAwMDApIj4KICAgICAgPGcgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMS40NjQwNTAsIDAuMzc4OTE0KSI+CiAgICAgICAgPHBhdGggZD0iTS00MTMuNywwdjU4LjNoNTJ2MjguMmgtNTJWMTk2YzAsMjUsNywzOS41LDI3LjMsMzkuNWM3LjEsMC4xLDE0LjItMC43LDIxLjEtMi41IGwxLjcsMjcuN2MtMTAuMywzLjctMjEuMyw1LjQtMzIuMiw1Yy03LjMsMC40LTE0LjYtMC43LTIxLjMtMy40Yy02LjgtMi43LTEyLjktNi44LTE3LjktMTIuMWMtMTAuMy0xMC45LTE0LjEtMjktMTQuMS01Mi45IFY4Ni41aC0zMVY1OC4zaDMxVjkuNkwtNDEzLjcsMEwtNDEzLjcsMHoiLz4KICAgICAgPC9nPgogICAgPC9nPgogICAgPGcgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoOTc0LjQzMzI4NiwgNTMuNDc5NjM4KSI+CiAgICAgIDxnIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAuOTkwMDM0LCAwLjYxMDMzOSkiPgogICAgICAgIDxwYXRoIGQ9Ik0tNDQ1LjgsMTEzYzAuOCw1MCwzMi4yLDcwLjYsNjguNiw3MC42YzE5LDAuNiwzNy45LTMsNTUuMy0xMC41bDYuMiwyNi40IGMtMjAuOSw4LjktNDMuNSwxMy4xLTY2LjIsMTIuNmMtNjEuNSwwLTk4LjMtNDEuMi05OC4zLTEwMi41Qy00ODAuMiw0OC4yLTQ0NC43LDAtMzg2LjUsMGM2NS4yLDAsODIuNyw1OC4zLDgyLjcsOTUuNyBjLTAuMSw1LjgtMC41LDExLjUtMS4yLDE3LjJoLTE0MC42SC00NDUuOEwtNDQ1LjgsMTEzeiBNLTMzOS4yLDg2LjZjMC40LTIzLjUtOS41LTYwLjEtNTAuNC02MC4xIGMtMzYuOCwwLTUyLjgsMzQuNC01NS43LDYwLjFILTMzOS4yTC0zMzkuMiw4Ni42TC0zMzkuMiw4Ni42eiIvPgogICAgICA8L2c+CiAgICA8L2c+CiAgICA8ZyB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxMjAxLjk2MTA1OCwgNTMuNDc5NjM4KSI+CiAgICAgIDxnIHRyYW5zZm9ybT0idHJhbnNsYXRlKDEuMTc5NjQwLCAwLjcwNTA2OCkiPgogICAgICAgIDxwYXRoIGQ9Ik0tNDc4LjYsNjhjMC0yMy45LTAuNC00NC41LTEuNy02My40aDMxLjhsMS4yLDM5LjloMS43YzkuMS0yNy4zLDMxLTQ0LjUsNTUuMy00NC41IGMzLjUtMC4xLDcsMC40LDEwLjMsMS4ydjM0LjhjLTQuMS0wLjktOC4yLTEuMy0xMi40LTEuMmMtMjUuNiwwLTQzLjcsMTkuNy00OC43LDQ3LjRjLTEsNS43LTEuNiwxMS41LTEuNywxNy4ydjEwOC4zaC0zNlY2OCBMLTQ3OC42LDY4eiIvPgogICAgICA8L2c+CiAgICA8L2c+CiAgPC9nPgoKICA8ZyBjbGFzcz0ianAtaWNvbi13YXJuMCIgZmlsbD0iI0YzNzcyNiI+CiAgICA8cGF0aCBkPSJNMTM1Mi4zLDMyNi4yaDM3VjI4aC0zN1YzMjYuMnogTTE2MDQuOCwzMjYuMmMtMi41LTEzLjktMy40LTMxLjEtMy40LTQ4Ljd2LTc2IGMwLTQwLjctMTUuMS04My4xLTc3LjMtODMuMWMtMjUuNiwwLTUwLDcuMS02Ni44LDE4LjFsOC40LDI0LjRjMTQuMy05LjIsMzQtMTUuMSw1My0xNS4xYzQxLjYsMCw0Ni4yLDMwLjIsNDYuMiw0N3Y0LjIgYy03OC42LTAuNC0xMjIuMywyNi41LTEyMi4zLDc1LjZjMCwyOS40LDIxLDU4LjQsNjIuMiw1OC40YzI5LDAsNTAuOS0xNC4zLDYyLjItMzAuMmgxLjNsMi45LDI1LjZIMTYwNC44eiBNMTU2NS43LDI1Ny43IGMwLDMuOC0wLjgsOC0yLjEsMTEuOGMtNS45LDE3LjItMjIuNywzNC00OS4yLDM0Yy0xOC45LDAtMzQuOS0xMS4zLTM0LjktMzUuM2MwLTM5LjUsNDUuOC00Ni42LDg2LjItNDUuOFYyNTcuN3ogTTE2OTguNSwzMjYuMiBsMS43LTMzLjZoMS4zYzE1LjEsMjYuOSwzOC43LDM4LjIsNjguMSwzOC4yYzQ1LjQsMCw5MS4yLTM2LjEsOTEuMi0xMDguOGMwLjQtNjEuNy0zNS4zLTEwMy43LTg1LjctMTAzLjcgYy0zMi44LDAtNTYuMywxNC43LTY5LjMsMzcuNGgtMC44VjI4aC0zNi42djI0NS43YzAsMTguMS0wLjgsMzguNi0xLjcsNTIuNUgxNjk4LjV6IE0xNzA0LjgsMjA4LjJjMC01LjksMS4zLTEwLjksMi4xLTE1LjEgYzcuNi0yOC4xLDMxLjEtNDUuNCw1Ni4zLTQ1LjRjMzkuNSwwLDYwLjUsMzQuOSw2MC41LDc1LjZjMCw0Ni42LTIzLjEsNzguMS02MS44LDc4LjFjLTI2LjksMC00OC4zLTE3LjYtNTUuNS00My4zIGMtMC44LTQuMi0xLjctOC44LTEuNy0xMy40VjIwOC4yeiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-kernel: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICAgIDxwYXRoIGNsYXNzPSJqcC1pY29uMiIgZmlsbD0iIzYxNjE2MSIgZD0iTTE1IDlIOXY2aDZWOXptLTIgNGgtMnYtMmgydjJ6bTgtMlY5aC0yVjdjMC0xLjEtLjktMi0yLTJoLTJWM2gtMnYyaC0yVjNIOXYySDdjLTEuMSAwLTIgLjktMiAydjJIM3YyaDJ2MkgzdjJoMnYyYzAgMS4xLjkgMiAyIDJoMnYyaDJ2LTJoMnYyaDJ2LTJoMmMxLjEgMCAyLS45IDItMnYtMmgydi0yaC0ydi0yaDJ6bS00IDZIN1Y3aDEwdjEweiIvPgo8L3N2Zz4K);
  --jp-icon-keyboard: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8cGF0aCBjbGFzcz0ianAtaWNvbjMganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjNjE2MTYxIiBkPSJNMjAgNUg0Yy0xLjEgMC0xLjk5LjktMS45OSAyTDIgMTdjMCAxLjEuOSAyIDIgMmgxNmMxLjEgMCAyLS45IDItMlY3YzAtMS4xLS45LTItMi0yem0tOSAzaDJ2MmgtMlY4em0wIDNoMnYyaC0ydi0yek04IDhoMnYySDhWOHptMCAzaDJ2Mkg4di0yem0tMSAySDV2LTJoMnYyem0wLTNINVY4aDJ2MnptOSA3SDh2LTJoOHYyem0wLTRoLTJ2LTJoMnYyem0wLTNoLTJWOGgydjJ6bTMgM2gtMnYtMmgydjJ6bTAtM2gtMlY4aDJ2MnoiLz4KPC9zdmc+Cg==);
  --jp-icon-launch: url(data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMzIgMzIiIHdpZHRoPSIzMiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjNjE2MTYxIj4KICAgIDxwYXRoIGQ9Ik0yNiwyOEg2YTIuMDAyNywyLjAwMjcsMCwwLDEtMi0yVjZBMi4wMDI3LDIuMDAyNywwLDAsMSw2LDRIMTZWNkg2VjI2SDI2VjE2aDJWMjZBMi4wMDI3LDIuMDAyNywwLDAsMSwyNiwyOFoiLz4KICAgIDxwb2x5Z29uIHBvaW50cz0iMjAgMiAyMCA0IDI2LjU4NiA0IDE4IDEyLjU4NiAxOS40MTQgMTQgMjggNS40MTQgMjggMTIgMzAgMTIgMzAgMiAyMCAyIi8+CiAgPC9nPgo8L3N2Zz4K);
  --jp-icon-launcher: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8cGF0aCBjbGFzcz0ianAtaWNvbjMganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjNjE2MTYxIiBkPSJNMTkgMTlINVY1aDdWM0g1YTIgMiAwIDAwLTIgMnYxNGEyIDIgMCAwMDIgMmgxNGMxLjEgMCAyLS45IDItMnYtN2gtMnY3ek0xNCAzdjJoMy41OWwtOS44MyA5LjgzIDEuNDEgMS40MUwxOSA2LjQxVjEwaDJWM2gtN3oiLz4KPC9zdmc+Cg==);
  --jp-icon-line-form: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICAgIDxwYXRoIGZpbGw9IndoaXRlIiBkPSJNNS44OCA0LjEyTDEzLjc2IDEybC03Ljg4IDcuODhMOCAyMmwxMC0xMEw4IDJ6Ii8+Cjwvc3ZnPgo=);
  --jp-icon-link: url(data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIxNiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTMuOSAxMmMwLTEuNzEgMS4zOS0zLjEgMy4xLTMuMWg0VjdIN2MtMi43NiAwLTUgMi4yNC01IDVzMi4yNCA1IDUgNWg0di0xLjlIN2MtMS43MSAwLTMuMS0xLjM5LTMuMS0zLjF6TTggMTNoOHYtMkg4djJ6bTktNmgtNHYxLjloNGMxLjcxIDAgMy4xIDEuMzkgMy4xIDMuMXMtMS4zOSAzLjEtMy4xIDMuMWgtNFYxN2g0YzIuNzYgMCA1LTIuMjQgNS01cy0yLjI0LTUtNS01eiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-list: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICAgIDxwYXRoIGNsYXNzPSJqcC1pY29uMiBqcC1pY29uLXNlbGVjdGFibGUiIGZpbGw9IiM2MTYxNjEiIGQ9Ik0xOSA1djE0SDVWNWgxNG0xLjEtMkgzLjljLS41IDAtLjkuNC0uOS45djE2LjJjMCAuNC40LjkuOS45aDE2LjJjLjQgMCAuOS0uNS45LS45VjMuOWMwLS41LS41LS45LS45LS45ek0xMSA3aDZ2MmgtNlY3em0wIDRoNnYyaC02di0yem0wIDRoNnYyaC02ek03IDdoMnYySDd6bTAgNGgydjJIN3ptMCA0aDJ2Mkg3eiIvPgo8L3N2Zz4K);
  --jp-icon-markdown: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIyIDIyIj4KICA8cGF0aCBjbGFzcz0ianAtaWNvbi1jb250cmFzdDAganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjN0IxRkEyIiBkPSJNNSAxNC45aDEybC02LjEgNnptOS40LTYuOGMwLTEuMy0uMS0yLjktLjEtNC41LS40IDEuNC0uOSAyLjktMS4zIDQuM2wtMS4zIDQuM2gtMkw4LjUgNy45Yy0uNC0xLjMtLjctMi45LTEtNC4zLS4xIDEuNi0uMSAzLjItLjIgNC42TDcgMTIuNEg0LjhsLjctMTFoMy4zTDEwIDVjLjQgMS4yLjcgMi43IDEgMy45LjMtMS4yLjctMi42IDEtMy45bDEuMi0zLjdoMy4zbC42IDExaC0yLjRsLS4zLTQuMnoiLz4KPC9zdmc+Cg==);
  --jp-icon-move-down: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTQiIGhlaWdodD0iMTQiIHZpZXdCb3g9IjAgMCAxNCAxNCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggY2xhc3M9ImpwLWljb24zIiBkPSJNMTIuNDcxIDcuNTI4OTlDMTIuNzYzMiA3LjIzNjg0IDEyLjc2MzIgNi43NjMxNiAxMi40NzEgNi40NzEwMVY2LjQ3MTAxQzEyLjE3OSA2LjE3OTA1IDExLjcwNTcgNi4xNzg4NCAxMS40MTM1IDYuNDcwNTRMNy43NSAxMC4xMjc1VjEuNzVDNy43NSAxLjMzNTc5IDcuNDE0MjEgMSA3IDFWMUM2LjU4NTc5IDEgNi4yNSAxLjMzNTc5IDYuMjUgMS43NVYxMC4xMjc1TDIuNTk3MjYgNi40NjgyMkMyLjMwMzM4IDYuMTczODEgMS44MjY0MSA2LjE3MzU5IDEuNTMyMjYgNi40Njc3NFY2LjQ2Nzc0QzEuMjM4MyA2Ljc2MTcgMS4yMzgzIDcuMjM4MyAxLjUzMjI2IDcuNTMyMjZMNi4yOTI4OSAxMi4yOTI5QzYuNjgzNDIgMTIuNjgzNCA3LjMxNjU4IDEyLjY4MzQgNy43MDcxMSAxMi4yOTI5TDEyLjQ3MSA3LjUyODk5WiIgZmlsbD0iIzYxNjE2MSIvPgo8L3N2Zz4K);
  --jp-icon-move-up: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTQiIGhlaWdodD0iMTQiIHZpZXdCb3g9IjAgMCAxNCAxNCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggY2xhc3M9ImpwLWljb24zIiBkPSJNMS41Mjg5OSA2LjQ3MTAxQzEuMjM2ODQgNi43NjMxNiAxLjIzNjg0IDcuMjM2ODQgMS41Mjg5OSA3LjUyODk5VjcuNTI4OTlDMS44MjA5NSA3LjgyMDk1IDIuMjk0MjYgNy44MjExNiAyLjU4NjQ5IDcuNTI5NDZMNi4yNSAzLjg3MjVWMTIuMjVDNi4yNSAxMi42NjQyIDYuNTg1NzkgMTMgNyAxM1YxM0M3LjQxNDIxIDEzIDcuNzUgMTIuNjY0MiA3Ljc1IDEyLjI1VjMuODcyNUwxMS40MDI3IDcuNTMxNzhDMTEuNjk2NiA3LjgyNjE5IDEyLjE3MzYgNy44MjY0MSAxMi40Njc3IDcuNTMyMjZWNy41MzIyNkMxMi43NjE3IDcuMjM4MyAxMi43NjE3IDYuNzYxNyAxMi40Njc3IDYuNDY3NzRMNy43MDcxMSAxLjcwNzExQzcuMzE2NTggMS4zMTY1OCA2LjY4MzQyIDEuMzE2NTggNi4yOTI4OSAxLjcwNzExTDEuNTI4OTkgNi40NzEwMVoiIGZpbGw9IiM2MTYxNjEiLz4KPC9zdmc+Cg==);
  --jp-icon-new-folder: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTIwIDZoLThsLTItMkg0Yy0xLjExIDAtMS45OS44OS0xLjk5IDJMMiAxOGMwIDEuMTEuODkgMiAyIDJoMTZjMS4xMSAwIDItLjg5IDItMlY4YzAtMS4xMS0uODktMi0yLTJ6bS0xIDhoLTN2M2gtMnYtM2gtM3YtMmgzVjloMnYzaDN2MnoiLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-not-trusted: url(data:image/svg+xml;base64,PHN2ZyBmaWxsPSJub25lIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI1IDI1Ij4KICAgIDxwYXRoIGNsYXNzPSJqcC1pY29uMiIgc3Ryb2tlPSIjMzMzMzMzIiBzdHJva2Utd2lkdGg9IjIiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDMgMykiIGQ9Ik0xLjg2MDk0IDExLjQ0MDlDMC44MjY0NDggOC43NzAyNyAwLjg2Mzc3OSA2LjA1NzY0IDEuMjQ5MDcgNC4xOTkzMkMyLjQ4MjA2IDMuOTMzNDcgNC4wODA2OCAzLjQwMzQ3IDUuNjAxMDIgMi44NDQ5QzcuMjM1NDkgMi4yNDQ0IDguODU2NjYgMS41ODE1IDkuOTg3NiAxLjA5NTM5QzExLjA1OTcgMS41ODM0MSAxMi42MDk0IDIuMjQ0NCAxNC4yMTggMi44NDMzOUMxNS43NTAzIDMuNDEzOTQgMTcuMzk5NSAzLjk1MjU4IDE4Ljc1MzkgNC4yMTM4NUMxOS4xMzY0IDYuMDcxNzcgMTkuMTcwOSA4Ljc3NzIyIDE4LjEzOSAxMS40NDA5QzE3LjAzMDMgMTQuMzAzMiAxNC42NjY4IDE3LjE4NDQgOS45OTk5OSAxOC45MzU0QzUuMzMzMTkgMTcuMTg0NCAyLjk2OTY4IDE0LjMwMzIgMS44NjA5NCAxMS40NDA5WiIvPgogICAgPHBhdGggY2xhc3M9ImpwLWljb24yIiBzdHJva2U9IiMzMzMzMzMiIHN0cm9rZS13aWR0aD0iMiIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoOS4zMTU5MiA5LjMyMDMxKSIgZD0iTTcuMzY4NDIgMEwwIDcuMzY0NzkiLz4KICAgIDxwYXRoIGNsYXNzPSJqcC1pY29uMiIgc3Ryb2tlPSIjMzMzMzMzIiBzdHJva2Utd2lkdGg9IjIiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDkuMzE1OTIgMTYuNjgzNikgc2NhbGUoMSAtMSkiIGQ9Ik03LjM2ODQyIDBMMCA3LjM2NDc5Ii8+Cjwvc3ZnPgo=);
  --jp-icon-notebook: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIyIDIyIj4KICA8ZyBjbGFzcz0ianAtbm90ZWJvb2staWNvbi1jb2xvciBqcC1pY29uLXNlbGVjdGFibGUiIGZpbGw9IiNFRjZDMDAiPgogICAgPHBhdGggZD0iTTE4LjcgMy4zdjE1LjRIMy4zVjMuM2gxNS40bTEuNS0xLjVIMS44djE4LjNoMTguM2wuMS0xOC4zeiIvPgogICAgPHBhdGggZD0iTTE2LjUgMTYuNWwtNS40LTQuMy01LjYgNC4zdi0xMWgxMXoiLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-numbering: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjIiIGhlaWdodD0iMjIiIHZpZXdCb3g9IjAgMCAyOCAyOCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KCTxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSI+CgkJPHBhdGggZD0iTTQgMTlINlYxOS41SDVWMjAuNUg2VjIxSDRWMjJIN1YxOEg0VjE5Wk01IDEwSDZWNkg0VjdINVYxMFpNNCAxM0g1LjhMNCAxNS4xVjE2SDdWMTVINS4yTDcgMTIuOVYxMkg0VjEzWk05IDdWOUgyM1Y3SDlaTTkgMjFIMjNWMTlIOVYyMVpNOSAxNUgyM1YxM0g5VjE1WiIvPgoJPC9nPgo8L3N2Zz4K);
  --jp-icon-offline-bolt: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgd2lkdGg9IjE2Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTEyIDIuMDJjLTUuNTEgMC05Ljk4IDQuNDctOS45OCA5Ljk4czQuNDcgOS45OCA5Ljk4IDkuOTggOS45OC00LjQ3IDkuOTgtOS45OFMxNy41MSAyLjAyIDEyIDIuMDJ6TTExLjQ4IDIwdi02LjI2SDhMMTMgNHY2LjI2aDMuMzVMMTEuNDggMjB6Ii8+CiAgPC9nPgo8L3N2Zz4K);
  --jp-icon-palette: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTE4IDEzVjIwSDRWNkg5LjAyQzkuMDcgNS4yOSA5LjI0IDQuNjIgOS41IDRINEMyLjkgNCAyIDQuOSAyIDZWMjBDMiAyMS4xIDIuOSAyMiA0IDIySDE4QzE5LjEgMjIgMjAgMjEuMSAyMCAyMFYxNUwxOCAxM1pNMTkuMyA4Ljg5QzE5Ljc0IDguMTkgMjAgNy4zOCAyMCA2LjVDMjAgNC4wMSAxNy45OSAyIDE1LjUgMkMxMy4wMSAyIDExIDQuMDEgMTEgNi41QzExIDguOTkgMTMuMDEgMTEgMTUuNDkgMTFDMTYuMzcgMTEgMTcuMTkgMTAuNzQgMTcuODggMTAuM0wyMSAxMy40MkwyMi40MiAxMkwxOS4zIDguODlaTTE1LjUgOUMxNC4xMiA5IDEzIDcuODggMTMgNi41QzEzIDUuMTIgMTQuMTIgNCAxNS41IDRDMTYuODggNCAxOCA1LjEyIDE4IDYuNUMxOCA3Ljg4IDE2Ljg4IDkgMTUuNSA5WiIvPgogICAgPHBhdGggZmlsbC1ydWxlPSJldmVub2RkIiBjbGlwLXJ1bGU9ImV2ZW5vZGQiIGQ9Ik00IDZIOS4wMTg5NEM5LjAwNjM5IDYuMTY1MDIgOSA2LjMzMTc2IDkgNi41QzkgOC44MTU3NyAxMC4yMTEgMTAuODQ4NyAxMi4wMzQzIDEySDlWMTRIMTZWMTIuOTgxMUMxNi41NzAzIDEyLjkzNzcgMTcuMTIgMTIuODIwNyAxNy42Mzk2IDEyLjYzOTZMMTggMTNWMjBINFY2Wk04IDhINlYxMEg4VjhaTTYgMTJIOFYxNEg2VjEyWk04IDE2SDZWMThIOFYxNlpNOSAxNkgxNlYxOEg5VjE2WiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-paste: url(data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjI0IiB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIyNCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICAgIDxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSI+CiAgICAgICAgPHBhdGggZD0iTTE5IDJoLTQuMThDMTQuNC44NCAxMy4zIDAgMTIgMGMtMS4zIDAtMi40Ljg0LTIuODIgMkg1Yy0xLjEgMC0yIC45LTIgMnYxNmMwIDEuMS45IDIgMiAyaDE0YzEuMSAwIDItLjkgMi0yVjRjMC0xLjEtLjktMi0yLTJ6bS03IDBjLjU1IDAgMSAuNDUgMSAxcy0uNDUgMS0xIDEtMS0uNDUtMS0xIC40NS0xIDEtMXptNyAxOEg1VjRoMnYzaDEwVjRoMnYxNnoiLz4KICAgIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-pdf: url(data:image/svg+xml;base64,PHN2ZwogICB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyMiAyMiIgd2lkdGg9IjE2Ij4KICAgIDxwYXRoIHRyYW5zZm9ybT0icm90YXRlKDQ1KSIgY2xhc3M9ImpwLWljb24tc2VsZWN0YWJsZSIgZmlsbD0iI0ZGMkEyQSIKICAgICAgIGQ9Im0gMjIuMzQ0MzY5LC0zLjAxNjM2NDIgaCA1LjYzODYwNCB2IDEuNTc5MjQzMyBoIC0zLjU0OTIyNyB2IDEuNTA4NjkyOTkgaCAzLjMzNzU3NiBWIDEuNjUwODE1NCBoIC0zLjMzNzU3NiB2IDMuNDM1MjYxMyBoIC0yLjA4OTM3NyB6IG0gLTcuMTM2NDQ0LDEuNTc5MjQzMyB2IDQuOTQzOTU0MyBoIDAuNzQ4OTIgcSAxLjI4MDc2MSwwIDEuOTUzNzAzLC0wLjYzNDk1MzUgMC42NzgzNjksLTAuNjM0OTUzNSAwLjY3ODM2OSwtMS44NDUxNjQxIDAsLTEuMjA0NzgzNTUgLTAuNjcyOTQyLC0xLjgzNDMxMDExIC0wLjY3Mjk0MiwtMC42Mjk1MjY1OSAtMS45NTkxMywtMC42Mjk1MjY1OSB6IG0gLTIuMDg5Mzc3LC0xLjU3OTI0MzMgaCAyLjIwMzM0MyBxIDEuODQ1MTY0LDAgMi43NDYwMzksMC4yNjU5MjA3IDAuOTA2MzAxLDAuMjYwNDkzNyAxLjU1MjEwOCwwLjg5MDAyMDMgMC41Njk4MywwLjU0ODEyMjMgMC44NDY2MDUsMS4yNjQ0ODAwNiAwLjI3Njc3NCwwLjcxNjM1NzgxIDAuMjc2Nzc0LDEuNjIyNjU4OTQgMCwwLjkxNzE1NTEgLTAuMjc2Nzc0LDEuNjM4OTM5OSAtMC4yNzY3NzUsMC43MTYzNTc4IC0wLjg0NjYwNSwxLjI2NDQ4IC0wLjY1MTIzNCwwLjYyOTUyNjYgLTEuNTYyOTYyLDAuODk1NDQ3MyAtMC45MTE3MjgsMC4yNjA0OTM3IC0yLjczNTE4NSwwLjI2MDQ5MzcgaCAtMi4yMDMzNDMgeiBtIC04LjE0NTg1NjUsMCBoIDMuNDY3ODIzIHEgMS41NDY2ODE2LDAgMi4zNzE1Nzg1LDAuNjg5MjIzIDAuODMwMzI0LDAuNjgzNzk2MSAwLjgzMDMyNCwxLjk1MzcwMzE0IDAsMS4yNzUzMzM5NyAtMC44MzAzMjQsMS45NjQ1NTcwNiBRIDkuOTg3MTk2MSwyLjI3NDkxNSA4LjQ0MDUxNDUsMi4yNzQ5MTUgSCA3LjA2MjA2ODQgViA1LjA4NjA3NjcgSCA0Ljk3MjY5MTUgWiBtIDIuMDg5Mzc2OSwxLjUxNDExOTkgdiAyLjI2MzAzOTQzIGggMS4xNTU5NDEgcSAwLjYwNzgxODgsMCAwLjkzODg2MjksLTAuMjkzMDU1NDcgMC4zMzEwNDQxLC0wLjI5ODQ4MjQxIDAuMzMxMDQ0MSwtMC44NDExNzc3MiAwLC0wLjU0MjY5NTMxIC0wLjMzMTA0NDEsLTAuODM1NzUwNzQgLTAuMzMxMDQ0MSwtMC4yOTMwNTU1IC0wLjkzODg2MjksLTAuMjkzMDU1NSB6IgovPgo8L3N2Zz4K);
  --jp-icon-python: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iLTEwIC0xMCAxMzEuMTYxMzYxNjk0MzM1OTQgMTMyLjM4ODk5OTkzODk2NDg0Ij4KICA8cGF0aCBjbGFzcz0ianAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjMzA2OTk4IiBkPSJNIDU0LjkxODc4NSw5LjE5Mjc0MjFlLTQgQyA1MC4zMzUxMzIsMC4wMjIyMTcyNyA0NS45NTc4NDYsMC40MTMxMzY5NyA0Mi4xMDYyODUsMS4wOTQ2NjkzIDMwLjc2MDA2OSwzLjA5OTE3MzEgMjguNzAwMDM2LDcuMjk0NzcxNCAyOC43MDAwMzUsMTUuMDMyMTY5IHYgMTAuMjE4NzUgaCAyNi44MTI1IHYgMy40MDYyNSBoIC0yNi44MTI1IC0xMC4wNjI1IGMgLTcuNzkyNDU5LDAgLTE0LjYxNTc1ODgsNC42ODM3MTcgLTE2Ljc0OTk5OTgsMTMuNTkzNzUgLTIuNDYxODE5OTgsMTAuMjEyOTY2IC0yLjU3MTAxNTA4LDE2LjU4NjAyMyAwLDI3LjI1IDEuOTA1OTI4Myw3LjkzNzg1MiA2LjQ1NzU0MzIsMTMuNTkzNzQ4IDE0LjI0OTk5OTgsMTMuNTkzNzUgaCA5LjIxODc1IHYgLTEyLjI1IGMgMCwtOC44NDk5MDIgNy42NTcxNDQsLTE2LjY1NjI0OCAxNi43NSwtMTYuNjU2MjUgaCAyNi43ODEyNSBjIDcuNDU0OTUxLDAgMTMuNDA2MjUzLC02LjEzODE2NCAxMy40MDYyNSwtMTMuNjI1IHYgLTI1LjUzMTI1IGMgMCwtNy4yNjYzMzg2IC02LjEyOTk4LC0xMi43MjQ3NzcxIC0xMy40MDYyNSwtMTMuOTM3NDk5NyBDIDY0LjI4MTU0OCwwLjMyNzk0Mzk3IDU5LjUwMjQzOCwtMC4wMjAzNzkwMyA1NC45MTg3ODUsOS4xOTI3NDIxZS00IFogbSAtMTQuNSw4LjIxODc1MDEyNTc5IGMgMi43Njk1NDcsMCA1LjAzMTI1LDIuMjk4NjQ1NiA1LjAzMTI1LDUuMTI0OTk5NiAtMmUtNiwyLjgxNjMzNiAtMi4yNjE3MDMsNS4wOTM3NSAtNS4wMzEyNSw1LjA5Mzc1IC0yLjc3OTQ3NiwtMWUtNiAtNS4wMzEyNSwtMi4yNzc0MTUgLTUuMDMxMjUsLTUuMDkzNzUgLTEwZS03LC0yLjgyNjM1MyAyLjI1MTc3NCwtNS4xMjQ5OTk2IDUuMDMxMjUsLTUuMTI0OTk5NiB6Ii8+CiAgPHBhdGggY2xhc3M9ImpwLWljb24tc2VsZWN0YWJsZSIgZmlsbD0iI2ZmZDQzYiIgZD0ibSA4NS42Mzc1MzUsMjguNjU3MTY5IHYgMTEuOTA2MjUgYyAwLDkuMjMwNzU1IC03LjgyNTg5NSwxNi45OTk5OTkgLTE2Ljc1LDE3IGggLTI2Ljc4MTI1IGMgLTcuMzM1ODMzLDAgLTEzLjQwNjI0OSw2LjI3ODQ4MyAtMTMuNDA2MjUsMTMuNjI1IHYgMjUuNTMxMjQ3IGMgMCw3LjI2NjM0NCA2LjMxODU4OCwxMS41NDAzMjQgMTMuNDA2MjUsMTMuNjI1MDA0IDguNDg3MzMxLDIuNDk1NjEgMTYuNjI2MjM3LDIuOTQ2NjMgMjYuNzgxMjUsMCA2Ljc1MDE1NSwtMS45NTQzOSAxMy40MDYyNTMsLTUuODg3NjEgMTMuNDA2MjUsLTEzLjYyNTAwNCBWIDg2LjUwMDkxOSBoIC0yNi43ODEyNSB2IC0zLjQwNjI1IGggMjYuNzgxMjUgMTMuNDA2MjU0IGMgNy43OTI0NjEsMCAxMC42OTYyNTEsLTUuNDM1NDA4IDEzLjQwNjI0MSwtMTMuNTkzNzUgMi43OTkzMywtOC4zOTg4ODYgMi42ODAyMiwtMTYuNDc1Nzc2IDAsLTI3LjI1IC0xLjkyNTc4LC03Ljc1NzQ0MSAtNS42MDM4NywtMTMuNTkzNzUgLTEzLjQwNjI0MSwtMTMuNTkzNzUgeiBtIC0xNS4wNjI1LDY0LjY1NjI1IGMgMi43Nzk0NzgsM2UtNiA1LjAzMTI1LDIuMjc3NDE3IDUuMDMxMjUsNS4wOTM3NDcgLTJlLTYsMi44MjYzNTQgLTIuMjUxNzc1LDUuMTI1MDA0IC01LjAzMTI1LDUuMTI1MDA0IC0yLjc2OTU1LDAgLTUuMDMxMjUsLTIuMjk4NjUgLTUuMDMxMjUsLTUuMTI1MDA0IDJlLTYsLTIuODE2MzMgMi4yNjE2OTcsLTUuMDkzNzQ3IDUuMDMxMjUsLTUuMDkzNzQ3IHoiLz4KPC9zdmc+Cg==);
  --jp-icon-r-kernel: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIyIDIyIj4KICA8cGF0aCBjbGFzcz0ianAtaWNvbi1jb250cmFzdDMganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjMjE5NkYzIiBkPSJNNC40IDIuNWMxLjItLjEgMi45LS4zIDQuOS0uMyAyLjUgMCA0LjEuNCA1LjIgMS4zIDEgLjcgMS41IDEuOSAxLjUgMy41IDAgMi0xLjQgMy41LTIuOSA0LjEgMS4yLjQgMS43IDEuNiAyLjIgMyAuNiAxLjkgMSAzLjkgMS4zIDQuNmgtMy44Yy0uMy0uNC0uOC0xLjctMS4yLTMuN3MtMS4yLTIuNi0yLjYtMi42aC0uOXY2LjRINC40VjIuNXptMy43IDYuOWgxLjRjMS45IDAgMi45LS45IDIuOS0yLjNzLTEtMi4zLTIuOC0yLjNjLS43IDAtMS4zIDAtMS42LjJ2NC41aC4xdi0uMXoiLz4KPC9zdmc+Cg==);
  --jp-icon-react: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMTUwIDE1MCA1NDEuOSAyOTUuMyI+CiAgPGcgY2xhc3M9ImpwLWljb24tYnJhbmQyIGpwLWljb24tc2VsZWN0YWJsZSIgZmlsbD0iIzYxREFGQiI+CiAgICA8cGF0aCBkPSJNNjY2LjMgMjk2LjVjMC0zMi41LTQwLjctNjMuMy0xMDMuMS04Mi40IDE0LjQtNjMuNiA4LTExNC4yLTIwLjItMTMwLjQtNi41LTMuOC0xNC4xLTUuNi0yMi40LTUuNnYyMi4zYzQuNiAwIDguMy45IDExLjQgMi42IDEzLjYgNy44IDE5LjUgMzcuNSAxNC45IDc1LjctMS4xIDkuNC0yLjkgMTkuMy01LjEgMjkuNC0xOS42LTQuOC00MS04LjUtNjMuNS0xMC45LTEzLjUtMTguNS0yNy41LTM1LjMtNDEuNi01MCAzMi42LTMwLjMgNjMuMi00Ni45IDg0LTQ2LjlWNzhjLTI3LjUgMC02My41IDE5LjYtOTkuOSA1My42LTM2LjQtMzMuOC03Mi40LTUzLjItOTkuOS01My4ydjIyLjNjMjAuNyAwIDUxLjQgMTYuNSA4NCA0Ni42LTE0IDE0LjctMjggMzEuNC00MS4zIDQ5LjktMjIuNiAyLjQtNDQgNi4xLTYzLjYgMTEtMi4zLTEwLTQtMTkuNy01LjItMjktNC43LTM4LjIgMS4xLTY3LjkgMTQuNi03NS44IDMtMS44IDYuOS0yLjYgMTEuNS0yLjZWNzguNWMtOC40IDAtMTYgMS44LTIyLjYgNS42LTI4LjEgMTYuMi0zNC40IDY2LjctMTkuOSAxMzAuMS02Mi4yIDE5LjItMTAyLjcgNDkuOS0xMDIuNyA4Mi4zIDAgMzIuNSA0MC43IDYzLjMgMTAzLjEgODIuNC0xNC40IDYzLjYtOCAxMTQuMiAyMC4yIDEzMC40IDYuNSAzLjggMTQuMSA1LjYgMjIuNSA1LjYgMjcuNSAwIDYzLjUtMTkuNiA5OS45LTUzLjYgMzYuNCAzMy44IDcyLjQgNTMuMiA5OS45IDUzLjIgOC40IDAgMTYtMS44IDIyLjYtNS42IDI4LjEtMTYuMiAzNC40LTY2LjcgMTkuOS0xMzAuMSA2Mi0xOS4xIDEwMi41LTQ5LjkgMTAyLjUtODIuM3ptLTEzMC4yLTY2LjdjLTMuNyAxMi45LTguMyAyNi4yLTEzLjUgMzkuNS00LjEtOC04LjQtMTYtMTMuMS0yNC00LjYtOC05LjUtMTUuOC0xNC40LTIzLjQgMTQuMiAyLjEgMjcuOSA0LjcgNDEgNy45em0tNDUuOCAxMDYuNWMtNy44IDEzLjUtMTUuOCAyNi4zLTI0LjEgMzguMi0xNC45IDEuMy0zMCAyLTQ1LjIgMi0xNS4xIDAtMzAuMi0uNy00NS0xLjktOC4zLTExLjktMTYuNC0yNC42LTI0LjItMzgtNy42LTEzLjEtMTQuNS0yNi40LTIwLjgtMzkuOCA2LjItMTMuNCAxMy4yLTI2LjggMjAuNy0zOS45IDcuOC0xMy41IDE1LjgtMjYuMyAyNC4xLTM4LjIgMTQuOS0xLjMgMzAtMiA0NS4yLTIgMTUuMSAwIDMwLjIuNyA0NSAxLjkgOC4zIDExLjkgMTYuNCAyNC42IDI0LjIgMzggNy42IDEzLjEgMTQuNSAyNi40IDIwLjggMzkuOC02LjMgMTMuNC0xMy4yIDI2LjgtMjAuNyAzOS45em0zMi4zLTEzYzUuNCAxMy40IDEwIDI2LjggMTMuOCAzOS44LTEzLjEgMy4yLTI2LjkgNS45LTQxLjIgOCA0LjktNy43IDkuOC0xNS42IDE0LjQtMjMuNyA0LjYtOCA4LjktMTYuMSAxMy0yNC4xek00MjEuMiA0MzBjLTkuMy05LjYtMTguNi0yMC4zLTI3LjgtMzIgOSAuNCAxOC4yLjcgMjcuNS43IDkuNCAwIDE4LjctLjIgMjcuOC0uNy05IDExLjctMTguMyAyMi40LTI3LjUgMzJ6bS03NC40LTU4LjljLTE0LjItMi4xLTI3LjktNC43LTQxLTcuOSAzLjctMTIuOSA4LjMtMjYuMiAxMy41LTM5LjUgNC4xIDggOC40IDE2IDEzLjEgMjQgNC43IDggOS41IDE1LjggMTQuNCAyMy40ek00MjAuNyAxNjNjOS4zIDkuNiAxOC42IDIwLjMgMjcuOCAzMi05LS40LTE4LjItLjctMjcuNS0uNy05LjQgMC0xOC43LjItMjcuOC43IDktMTEuNyAxOC4zLTIyLjQgMjcuNS0zMnptLTc0IDU4LjljLTQuOSA3LjctOS44IDE1LjYtMTQuNCAyMy43LTQuNiA4LTguOSAxNi0xMyAyNC01LjQtMTMuNC0xMC0yNi44LTEzLjgtMzkuOCAxMy4xLTMuMSAyNi45LTUuOCA0MS4yLTcuOXptLTkwLjUgMTI1LjJjLTM1LjQtMTUuMS01OC4zLTM0LjktNTguMy01MC42IDAtMTUuNyAyMi45LTM1LjYgNTguMy01MC42IDguNi0zLjcgMTgtNyAyNy43LTEwLjEgNS43IDE5LjYgMTMuMiA0MCAyMi41IDYwLjktOS4yIDIwLjgtMTYuNiA0MS4xLTIyLjIgNjAuNi05LjktMy4xLTE5LjMtNi41LTI4LTEwLjJ6TTMxMCA0OTBjLTEzLjYtNy44LTE5LjUtMzcuNS0xNC45LTc1LjcgMS4xLTkuNCAyLjktMTkuMyA1LjEtMjkuNCAxOS42IDQuOCA0MSA4LjUgNjMuNSAxMC45IDEzLjUgMTguNSAyNy41IDM1LjMgNDEuNiA1MC0zMi42IDMwLjMtNjMuMiA0Ni45LTg0IDQ2LjktNC41LS4xLTguMy0xLTExLjMtMi43em0yMzcuMi03Ni4yYzQuNyAzOC4yLTEuMSA2Ny45LTE0LjYgNzUuOC0zIDEuOC02LjkgMi42LTExLjUgMi42LTIwLjcgMC01MS40LTE2LjUtODQtNDYuNiAxNC0xNC43IDI4LTMxLjQgNDEuMy00OS45IDIyLjYtMi40IDQ0LTYuMSA2My42LTExIDIuMyAxMC4xIDQuMSAxOS44IDUuMiAyOS4xem0zOC41LTY2LjdjLTguNiAzLjctMTggNy0yNy43IDEwLjEtNS43LTE5LjYtMTMuMi00MC0yMi41LTYwLjkgOS4yLTIwLjggMTYuNi00MS4xIDIyLjItNjAuNiA5LjkgMy4xIDE5LjMgNi41IDI4LjEgMTAuMiAzNS40IDE1LjEgNTguMyAzNC45IDU4LjMgNTAuNi0uMSAxNS43LTIzIDM1LjYtNTguNCA1MC42ek0zMjAuOCA3OC40eiIvPgogICAgPGNpcmNsZSBjeD0iNDIwLjkiIGN5PSIyOTYuNSIgcj0iNDUuNyIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-redo: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIGhlaWdodD0iMjQiIHZpZXdCb3g9IjAgMCAyNCAyNCIgd2lkdGg9IjE2Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgICA8cGF0aCBkPSJNMCAwaDI0djI0SDB6IiBmaWxsPSJub25lIi8+PHBhdGggZD0iTTE4LjQgMTAuNkMxNi41NSA4Ljk5IDE0LjE1IDggMTEuNSA4Yy00LjY1IDAtOC41OCAzLjAzLTkuOTYgNy4yMkwzLjkgMTZjMS4wNS0zLjE5IDQuMDUtNS41IDcuNi01LjUgMS45NSAwIDMuNzMuNzIgNS4xMiAxLjg4TDEzIDE2aDlWN2wtMy42IDMuNnoiLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-refresh: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDE4IDE4Ij4KICAgIDxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSI+CiAgICAgICAgPHBhdGggZD0iTTkgMTMuNWMtMi40OSAwLTQuNS0yLjAxLTQuNS00LjVTNi41MSA0LjUgOSA0LjVjMS4yNCAwIDIuMzYuNTIgMy4xNyAxLjMzTDEwIDhoNVYzbC0xLjc2IDEuNzZDMTIuMTUgMy42OCAxMC42NiAzIDkgMyA1LjY5IDMgMy4wMSA1LjY5IDMuMDEgOVM1LjY5IDE1IDkgMTVjMi45NyAwIDUuNDMtMi4xNiA1LjktNWgtMS41MmMtLjQ2IDItMi4yNCAzLjUtNC4zOCAzLjV6Ii8+CiAgICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-regex: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIwIDIwIj4KICA8ZyBjbGFzcz0ianAtaWNvbjIiIGZpbGw9IiM0MTQxNDEiPgogICAgPHJlY3QgeD0iMiIgeT0iMiIgd2lkdGg9IjE2IiBoZWlnaHQ9IjE2Ii8+CiAgPC9nPgoKICA8ZyBjbGFzcz0ianAtaWNvbi1hY2NlbnQyIiBmaWxsPSIjRkZGIj4KICAgIDxjaXJjbGUgY2xhc3M9InN0MiIgY3g9IjUuNSIgY3k9IjE0LjUiIHI9IjEuNSIvPgogICAgPHJlY3QgeD0iMTIiIHk9IjQiIGNsYXNzPSJzdDIiIHdpZHRoPSIxIiBoZWlnaHQ9IjgiLz4KICAgIDxyZWN0IHg9IjguNSIgeT0iNy41IiB0cmFuc2Zvcm09Im1hdHJpeCgwLjg2NiAtMC41IDAuNSAwLjg2NiAtMi4zMjU1IDcuMzIxOSkiIGNsYXNzPSJzdDIiIHdpZHRoPSI4IiBoZWlnaHQ9IjEiLz4KICAgIDxyZWN0IHg9IjEyIiB5PSI0IiB0cmFuc2Zvcm09Im1hdHJpeCgwLjUgLTAuODY2IDAuODY2IDAuNSAtMC42Nzc5IDE0LjgyNTIpIiBjbGFzcz0ic3QyIiB3aWR0aD0iMSIgaGVpZ2h0PSI4Ii8+CiAgPC9nPgo8L3N2Zz4K);
  --jp-icon-run: url(data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjI0IiB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIyNCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICAgIDxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSI+CiAgICAgICAgPHBhdGggZD0iTTggNXYxNGwxMS03eiIvPgogICAgPC9nPgo8L3N2Zz4K);
  --jp-icon-running: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDUxMiA1MTIiPgogIDxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSI+CiAgICA8cGF0aCBkPSJNMjU2IDhDMTE5IDggOCAxMTkgOCAyNTZzMTExIDI0OCAyNDggMjQ4IDI0OC0xMTEgMjQ4LTI0OFMzOTMgOCAyNTYgOHptOTYgMzI4YzAgOC44LTcuMiAxNi0xNiAxNkgxNzZjLTguOCAwLTE2LTcuMi0xNi0xNlYxNzZjMC04LjggNy4yLTE2IDE2LTE2aDE2MGM4LjggMCAxNiA3LjIgMTYgMTZ2MTYweiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-save: url(data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjI0IiB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIyNCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICAgIDxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSI+CiAgICAgICAgPHBhdGggZD0iTTE3IDNINWMtMS4xMSAwLTIgLjktMiAydjE0YzAgMS4xLjg5IDIgMiAyaDE0YzEuMSAwIDItLjkgMi0yVjdsLTQtNHptLTUgMTZjLTEuNjYgMC0zLTEuMzQtMy0zczEuMzQtMyAzLTMgMyAxLjM0IDMgMy0xLjM0IDMtMyAzem0zLTEwSDVWNWgxMHY0eiIvPgogICAgPC9nPgo8L3N2Zz4K);
  --jp-icon-search: url(data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMTggMTgiIHdpZHRoPSIxNiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTEyLjEsMTAuOWgtMC43bC0wLjItMC4yYzAuOC0wLjksMS4zLTIuMiwxLjMtMy41YzAtMy0yLjQtNS40LTUuNC01LjRTMS44LDQuMiwxLjgsNy4xczIuNCw1LjQsNS40LDUuNCBjMS4zLDAsMi41LTAuNSwzLjUtMS4zbDAuMiwwLjJ2MC43bDQuMSw0LjFsMS4yLTEuMkwxMi4xLDEwLjl6IE03LjEsMTAuOWMtMi4xLDAtMy43LTEuNy0zLjctMy43czEuNy0zLjcsMy43LTMuN3MzLjcsMS43LDMuNywzLjcgUzkuMiwxMC45LDcuMSwxMC45eiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-settings: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8cGF0aCBjbGFzcz0ianAtaWNvbjMganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjNjE2MTYxIiBkPSJNMTkuNDMgMTIuOThjLjA0LS4zMi4wNy0uNjQuMDctLjk4cy0uMDMtLjY2LS4wNy0uOThsMi4xMS0xLjY1Yy4xOS0uMTUuMjQtLjQyLjEyLS42NGwtMi0zLjQ2Yy0uMTItLjIyLS4zOS0uMy0uNjEtLjIybC0yLjQ5IDFjLS41Mi0uNC0xLjA4LS43My0xLjY5LS45OGwtLjM4LTIuNjVBLjQ4OC40ODggMCAwMDE0IDJoLTRjLS4yNSAwLS40Ni4xOC0uNDkuNDJsLS4zOCAyLjY1Yy0uNjEuMjUtMS4xNy41OS0xLjY5Ljk4bC0yLjQ5LTFjLS4yMy0uMDktLjQ5IDAtLjYxLjIybC0yIDMuNDZjLS4xMy4yMi0uMDcuNDkuMTIuNjRsMi4xMSAxLjY1Yy0uMDQuMzItLjA3LjY1LS4wNy45OHMuMDMuNjYuMDcuOThsLTIuMTEgMS42NWMtLjE5LjE1LS4yNC40Mi0uMTIuNjRsMiAzLjQ2Yy4xMi4yMi4zOS4zLjYxLjIybDIuNDktMWMuNTIuNCAxLjA4LjczIDEuNjkuOThsLjM4IDIuNjVjLjAzLjI0LjI0LjQyLjQ5LjQyaDRjLjI1IDAgLjQ2LS4xOC40OS0uNDJsLjM4LTIuNjVjLjYxLS4yNSAxLjE3LS41OSAxLjY5LS45OGwyLjQ5IDFjLjIzLjA5LjQ5IDAgLjYxLS4yMmwyLTMuNDZjLjEyLS4yMi4wNy0uNDktLjEyLS42NGwtMi4xMS0xLjY1ek0xMiAxNS41Yy0xLjkzIDAtMy41LTEuNTctMy41LTMuNXMxLjU3LTMuNSAzLjUtMy41IDMuNSAxLjU3IDMuNSAzLjUtMS41NyAzLjUtMy41IDMuNXoiLz4KPC9zdmc+Cg==);
  --jp-icon-share: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTYiIHZpZXdCb3g9IjAgMCAyNCAyNCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTSAxOCAyIEMgMTYuMzU0OTkgMiAxNSAzLjM1NDk5MDQgMTUgNSBDIDE1IDUuMTkwOTUyOSAxNS4wMjE3OTEgNS4zNzcxMjI0IDE1LjA1NjY0MSA1LjU1ODU5MzggTCA3LjkyMTg3NSA5LjcyMDcwMzEgQyA3LjM5ODUzOTkgOS4yNzc4NTM5IDYuNzMyMDc3MSA5IDYgOSBDIDQuMzU0OTkwNCA5IDMgMTAuMzU0OTkgMyAxMiBDIDMgMTMuNjQ1MDEgNC4zNTQ5OTA0IDE1IDYgMTUgQyA2LjczMjA3NzEgMTUgNy4zOTg1Mzk5IDE0LjcyMjE0NiA3LjkyMTg3NSAxNC4yNzkyOTcgTCAxNS4wNTY2NDEgMTguNDM5NDUzIEMgMTUuMDIxNTU1IDE4LjYyMTUxNCAxNSAxOC44MDgzODYgMTUgMTkgQyAxNSAyMC42NDUwMSAxNi4zNTQ5OSAyMiAxOCAyMiBDIDE5LjY0NTAxIDIyIDIxIDIwLjY0NTAxIDIxIDE5IEMgMjEgMTcuMzU0OTkgMTkuNjQ1MDEgMTYgMTggMTYgQyAxNy4yNjc0OCAxNiAxNi42MDE1OTMgMTYuMjc5MzI4IDE2LjA3ODEyNSAxNi43MjI2NTYgTCA4Ljk0MzM1OTQgMTIuNTU4NTk0IEMgOC45NzgyMDk1IDEyLjM3NzEyMiA5IDEyLjE5MDk1MyA5IDEyIEMgOSAxMS44MDkwNDcgOC45NzgyMDk1IDExLjYyMjg3OCA4Ljk0MzM1OTQgMTEuNDQxNDA2IEwgMTYuMDc4MTI1IDcuMjc5Mjk2OSBDIDE2LjYwMTQ2IDcuNzIyMTQ2MSAxNy4yNjc5MjMgOCAxOCA4IEMgMTkuNjQ1MDEgOCAyMSA2LjY0NTAwOTYgMjEgNSBDIDIxIDMuMzU0OTkwNCAxOS42NDUwMSAyIDE4IDIgeiBNIDE4IDQgQyAxOC41NjQxMjkgNCAxOSA0LjQzNTg3MDYgMTkgNSBDIDE5IDUuNTY0MTI5NCAxOC41NjQxMjkgNiAxOCA2IEMgMTcuNDM1ODcxIDYgMTcgNS41NjQxMjk0IDE3IDUgQyAxNyA0LjQzNTg3MDYgMTcuNDM1ODcxIDQgMTggNCB6IE0gNiAxMSBDIDYuNTY0MTI5NCAxMSA3IDExLjQzNTg3MSA3IDEyIEMgNyAxMi41NjQxMjkgNi41NjQxMjk0IDEzIDYgMTMgQyA1LjQzNTg3MDYgMTMgNSAxMi41NjQxMjkgNSAxMiBDIDUgMTEuNDM1ODcxIDUuNDM1ODcwNiAxMSA2IDExIHogTSAxOCAxOCBDIDE4LjU2NDEyOSAxOCAxOSAxOC40MzU4NzEgMTkgMTkgQyAxOSAxOS41NjQxMjkgMTguNTY0MTI5IDIwIDE4IDIwIEMgMTcuNDM1ODcxIDIwIDE3IDE5LjU2NDEyOSAxNyAxOSBDIDE3IDE4LjQzNTg3MSAxNy40MzU4NzEgMTggMTggMTggeiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-spreadsheet: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIyIDIyIj4KICA8cGF0aCBjbGFzcz0ianAtaWNvbi1jb250cmFzdDEganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjNENBRjUwIiBkPSJNMi4yIDIuMnYxNy42aDE3LjZWMi4ySDIuMnptMTUuNCA3LjdoLTUuNVY0LjRoNS41djUuNXpNOS45IDQuNHY1LjVINC40VjQuNGg1LjV6bS01LjUgNy43aDUuNXY1LjVINC40di01LjV6bTcuNyA1LjV2LTUuNWg1LjV2NS41aC01LjV6Ii8+Cjwvc3ZnPgo=);
  --jp-icon-stop: url(data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjI0IiB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIyNCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICAgIDxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSI+CiAgICAgICAgPHBhdGggZD0iTTAgMGgyNHYyNEgweiIgZmlsbD0ibm9uZSIvPgogICAgICAgIDxwYXRoIGQ9Ik02IDZoMTJ2MTJINnoiLz4KICAgIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-tab: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTIxIDNIM2MtMS4xIDAtMiAuOS0yIDJ2MTRjMCAxLjEuOSAyIDIgMmgxOGMxLjEgMCAyLS45IDItMlY1YzAtMS4xLS45LTItMi0yem0wIDE2SDNWNWgxMHY0aDh2MTB6Ii8+CiAgPC9nPgo8L3N2Zz4K);
  --jp-icon-table-rows: url(data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjI0IiB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIyNCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICAgIDxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSI+CiAgICAgICAgPHBhdGggZD0iTTAgMGgyNHYyNEgweiIgZmlsbD0ibm9uZSIvPgogICAgICAgIDxwYXRoIGQ9Ik0yMSw4SDNWNGgxOFY4eiBNMjEsMTBIM3Y0aDE4VjEweiBNMjEsMTZIM3Y0aDE4VjE2eiIvPgogICAgPC9nPgo8L3N2Zz4K);
  --jp-icon-tag: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjgiIGhlaWdodD0iMjgiIHZpZXdCb3g9IjAgMCA0MyAyOCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KCTxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSI+CgkJPHBhdGggZD0iTTI4LjgzMzIgMTIuMzM0TDMyLjk5OTggMTYuNTAwN0wzNy4xNjY1IDEyLjMzNEgyOC44MzMyWiIvPgoJCTxwYXRoIGQ9Ik0xNi4yMDk1IDIxLjYxMDRDMTUuNjg3MyAyMi4xMjk5IDE0Ljg0NDMgMjIuMTI5OSAxNC4zMjQ4IDIxLjYxMDRMNi45ODI5IDE0LjcyNDVDNi41NzI0IDE0LjMzOTQgNi4wODMxMyAxMy42MDk4IDYuMDQ3ODYgMTMuMDQ4MkM1Ljk1MzQ3IDExLjUyODggNi4wMjAwMiA4LjYxOTQ0IDYuMDY2MjEgNy4wNzY5NUM2LjA4MjgxIDYuNTE0NzcgNi41NTU0OCA2LjA0MzQ3IDcuMTE4MDQgNi4wMzA1NUM5LjA4ODYzIDUuOTg0NzMgMTMuMjYzOCA1LjkzNTc5IDEzLjY1MTggNi4zMjQyNUwyMS43MzY5IDEzLjYzOUMyMi4yNTYgMTQuMTU4NSAyMS43ODUxIDE1LjQ3MjQgMjEuMjYyIDE1Ljk5NDZMMTYuMjA5NSAyMS42MTA0Wk05Ljc3NTg1IDguMjY1QzkuMzM1NTEgNy44MjU2NiA4LjYyMzUxIDcuODI1NjYgOC4xODI4IDguMjY1QzcuNzQzNDYgOC43MDU3MSA3Ljc0MzQ2IDkuNDE3MzMgOC4xODI4IDkuODU2NjdDOC42MjM4MiAxMC4yOTY0IDkuMzM1ODIgMTAuMjk2NCA5Ljc3NTg1IDkuODU2NjdDMTAuMjE1NiA5LjQxNzMzIDEwLjIxNTYgOC43MDUzMyA5Ljc3NTg1IDguMjY1WiIvPgoJPC9nPgo8L3N2Zz4K);
  --jp-icon-terminal: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0IiA+CiAgICA8cmVjdCBjbGFzcz0ianAtdGVybWluYWwtaWNvbi1iYWNrZ3JvdW5kLWNvbG9yIGpwLWljb24tc2VsZWN0YWJsZSIgd2lkdGg9IjIwIiBoZWlnaHQ9IjIwIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgyIDIpIiBmaWxsPSIjMzMzMzMzIi8+CiAgICA8cGF0aCBjbGFzcz0ianAtdGVybWluYWwtaWNvbi1jb2xvciBqcC1pY29uLXNlbGVjdGFibGUtaW52ZXJzZSIgZD0iTTUuMDU2NjQgOC43NjE3MkM1LjA1NjY0IDguNTk3NjYgNS4wMzEyNSA4LjQ1MzEyIDQuOTgwNDcgOC4zMjgxMkM0LjkzMzU5IDguMTk5MjIgNC44NTU0NyA4LjA4MjAzIDQuNzQ2MDkgNy45NzY1NkM0LjY0MDYyIDcuODcxMDkgNC41IDcuNzc1MzkgNC4zMjQyMiA3LjY4OTQ1QzQuMTUyMzQgNy41OTk2MSAzLjk0MzM2IDcuNTExNzIgMy42OTcyNyA3LjQyNTc4QzMuMzAyNzMgNy4yODUxNiAyLjk0MzM2IDcuMTM2NzIgMi42MTkxNCA2Ljk4MDQ3QzIuMjk0OTIgNi44MjQyMiAyLjAxNzU4IDYuNjQyNTggMS43ODcxMSA2LjQzNTU1QzEuNTYwNTUgNi4yMjg1MiAxLjM4NDc3IDUuOTg4MjggMS4yNTk3NyA1LjcxNDg0QzEuMTM0NzcgNS40Mzc1IDEuMDcyMjcgNS4xMDkzOCAxLjA3MjI3IDQuNzMwNDdDMS4wNzIyNyA0LjM5ODQ0IDEuMTI4OTEgNC4wOTU3IDEuMjQyMTkgMy44MjIyN0MxLjM1NTQ3IDMuNTQ0OTIgMS41MTU2MiAzLjMwNDY5IDEuNzIyNjYgMy4xMDE1NkMxLjkyOTY5IDIuODk4NDQgMi4xNzk2OSAyLjczNDM3IDIuNDcyNjYgMi42MDkzOEMyLjc2NTYyIDIuNDg0MzggMy4wOTE4IDIuNDA0MyAzLjQ1MTE3IDIuMzY5MTRWMS4xMDkzOEg0LjM4ODY3VjIuMzgwODZDNC43NDAyMyAyLjQyNzczIDUuMDU2NjQgMi41MjM0NCA1LjMzNzg5IDIuNjY3OTdDNS42MTkxNCAyLjgxMjUgNS44NTc0MiAzLjAwMTk1IDYuMDUyNzMgMy4yMzYzM0M2LjI1MTk1IDMuNDY2OCA2LjQwNDMgMy43NDAyMyA2LjUwOTc3IDQuMDU2NjRDNi42MTkxNCA0LjM2OTE0IDYuNjczODMgNC43MjA3IDYuNjczODMgNS4xMTEzM0g1LjA0NDkyQzUuMDQ0OTIgNC42Mzg2NyA0LjkzNzUgNC4yODEyNSA0LjcyMjY2IDQuMDM5MDZDNC41MDc4MSAzLjc5Mjk3IDQuMjE2OCAzLjY2OTkyIDMuODQ5NjEgMy42Njk5MkMzLjY1MDM5IDMuNjY5OTIgMy40NzY1NiAzLjY5NzI3IDMuMzI4MTIgMy43NTE5NUMzLjE4MzU5IDMuODAyNzMgMy4wNjQ0NSAzLjg3Njk1IDIuOTcwNyAzLjk3NDYxQzIuODc2OTUgNC4wNjgzNiAyLjgwNjY0IDQuMTc5NjkgMi43NTk3NyA0LjMwODU5QzIuNzE2OCA0LjQzNzUgMi42OTUzMSA0LjU3ODEyIDIuNjk1MzEgNC43MzA0N0MyLjY5NTMxIDQuODgyODEgMi43MTY4IDUuMDE5NTMgMi43NTk3NyA1LjE0MDYyQzIuODA2NjQgNS4yNTc4MSAyLjg4MjgxIDUuMzY3MTkgMi45ODgyOCA1LjQ2ODc1QzMuMDk3NjYgNS41NzAzMSAzLjI0MDIzIDUuNjY3OTcgMy40MTYwMiA1Ljc2MTcyQzMuNTkxOCA1Ljg1MTU2IDMuODEwNTUgNS45NDMzNiA0LjA3MjI3IDYuMDM3MTFDNC40NjY4IDYuMTg1NTUgNC44MjQyMiA2LjMzOTg0IDUuMTQ0NTMgNi41QzUuNDY0ODQgNi42NTYyNSA1LjczODI4IDYuODM5ODQgNS45NjQ4NCA3LjA1MDc4QzYuMTk1MzEgNy4yNTc4MSA2LjM3MTA5IDcuNSA2LjQ5MjE5IDcuNzc3MzRDNi42MTcxOSA4LjA1MDc4IDYuNjc5NjkgOC4zNzUgNi42Nzk2OSA4Ljc1QzYuNjc5NjkgOS4wOTM3NSA2LjYyMzA1IDkuNDA0MyA2LjUwOTc3IDkuNjgxNjRDNi4zOTY0OCA5Ljk1NTA4IDYuMjM0MzggMTAuMTkxNCA2LjAyMzQ0IDEwLjM5MDZDNS44MTI1IDEwLjU4OTggNS41NTg1OSAxMC43NSA1LjI2MTcyIDEwLjg3MTFDNC45NjQ4NCAxMC45ODgzIDQuNjMyODEgMTEuMDY0NSA0LjI2NTYyIDExLjA5OTZWMTIuMjQ4SDMuMzMzOThWMTEuMDk5NkMzLjAwMTk1IDExLjA2ODQgMi42Nzk2OSAxMC45OTYxIDIuMzY3MTkgMTAuODgyOEMyLjA1NDY5IDEwLjc2NTYgMS43NzczNCAxMC41OTc3IDEuNTM1MTYgMTAuMzc4OUMxLjI5Njg4IDEwLjE2MDIgMS4xMDU0NyA5Ljg4NDc3IDAuOTYwOTM4IDkuNTUyNzNDMC44MTY0MDYgOS4yMTY4IDAuNzQ0MTQxIDguODE0NDUgMC43NDQxNDEgOC4zNDU3SDIuMzc4OTFDMi4zNzg5MSA4LjYyNjk1IDIuNDE5OTIgOC44NjMyOCAyLjUwMTk1IDkuMDU0NjlDMi41ODM5OCA5LjI0MjE5IDIuNjg5NDUgOS4zOTI1OCAyLjgxODM2IDkuNTA1ODZDMi45NTExNyA5LjYxNTIzIDMuMTAxNTYgOS42OTMzNiAzLjI2OTUzIDkuNzQwMjNDMy40Mzc1IDkuNzg3MTEgMy42MDkzOCA5LjgxMDU1IDMuNzg1MTYgOS44MTA1NUM0LjIwMzEyIDkuODEwNTUgNC41MTk1MyA5LjcxMjg5IDQuNzM0MzggOS41MTc1OEM0Ljk0OTIyIDkuMzIyMjcgNS4wNTY2NCA5LjA3MDMxIDUuMDU2NjQgOC43NjE3MlpNMTMuNDE4IDEyLjI3MTVIOC4wNzQyMlYxMUgxMy40MThWMTIuMjcxNVoiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDMuOTUyNjQgNikiIGZpbGw9IndoaXRlIi8+Cjwvc3ZnPgo=);
  --jp-icon-text-editor: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8cGF0aCBjbGFzcz0ianAtdGV4dC1lZGl0b3ItaWNvbi1jb2xvciBqcC1pY29uLXNlbGVjdGFibGUiIGZpbGw9IiM2MTYxNjEiIGQ9Ik0xNSAxNUgzdjJoMTJ2LTJ6bTAtOEgzdjJoMTJWN3pNMyAxM2gxOHYtMkgzdjJ6bTAgOGgxOHYtMkgzdjJ6TTMgM3YyaDE4VjNIM3oiLz4KPC9zdmc+Cg==);
  --jp-icon-toc: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjNjE2MTYxIj4KICAgIDxwYXRoIGQ9Ik03LDVIMjFWN0g3VjVNNywxM1YxMUgyMVYxM0g3TTQsNC41QTEuNSwxLjUgMCAwLDEgNS41LDZBMS41LDEuNSAwIDAsMSA0LDcuNUExLjUsMS41IDAgMCwxIDIuNSw2QTEuNSwxLjUgMCAwLDEgNCw0LjVNNCwxMC41QTEuNSwxLjUgMCAwLDEgNS41LDEyQTEuNSwxLjUgMCAwLDEgNCwxMy41QTEuNSwxLjUgMCAwLDEgMi41LDEyQTEuNSwxLjUgMCAwLDEgNCwxMC41TTcsMTlWMTdIMjFWMTlIN000LDE2LjVBMS41LDEuNSAwIDAsMSA1LjUsMThBMS41LDEuNSAwIDAsMSA0LDE5LjVBMS41LDEuNSAwIDAsMSAyLjUsMThBMS41LDEuNSAwIDAsMSA0LDE2LjVaIiAvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-tree-view: url(data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjI0IiB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIyNCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICAgIDxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSI+CiAgICAgICAgPHBhdGggZD0iTTAgMGgyNHYyNEgweiIgZmlsbD0ibm9uZSIvPgogICAgICAgIDxwYXRoIGQ9Ik0yMiAxMVYzaC03djNIOVYzSDJ2OGg3VjhoMnYxMGg0djNoN3YtOGgtN3YzaC0yVjhoMnYzeiIvPgogICAgPC9nPgo8L3N2Zz4K);
  --jp-icon-trusted: url(data:image/svg+xml;base64,PHN2ZyBmaWxsPSJub25lIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI1Ij4KICAgIDxwYXRoIGNsYXNzPSJqcC1pY29uMiIgc3Ryb2tlPSIjMzMzMzMzIiBzdHJva2Utd2lkdGg9IjIiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDIgMykiIGQ9Ik0xLjg2MDk0IDExLjQ0MDlDMC44MjY0NDggOC43NzAyNyAwLjg2Mzc3OSA2LjA1NzY0IDEuMjQ5MDcgNC4xOTkzMkMyLjQ4MjA2IDMuOTMzNDcgNC4wODA2OCAzLjQwMzQ3IDUuNjAxMDIgMi44NDQ5QzcuMjM1NDkgMi4yNDQ0IDguODU2NjYgMS41ODE1IDkuOTg3NiAxLjA5NTM5QzExLjA1OTcgMS41ODM0MSAxMi42MDk0IDIuMjQ0NCAxNC4yMTggMi44NDMzOUMxNS43NTAzIDMuNDEzOTQgMTcuMzk5NSAzLjk1MjU4IDE4Ljc1MzkgNC4yMTM4NUMxOS4xMzY0IDYuMDcxNzcgMTkuMTcwOSA4Ljc3NzIyIDE4LjEzOSAxMS40NDA5QzE3LjAzMDMgMTQuMzAzMiAxNC42NjY4IDE3LjE4NDQgOS45OTk5OSAxOC45MzU0QzUuMzMzMiAxNy4xODQ0IDIuOTY5NjggMTQuMzAzMiAxLjg2MDk0IDExLjQ0MDlaIi8+CiAgICA8cGF0aCBjbGFzcz0ianAtaWNvbjIiIGZpbGw9IiMzMzMzMzMiIHN0cm9rZT0iIzMzMzMzMyIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoOCA5Ljg2NzE5KSIgZD0iTTIuODYwMTUgNC44NjUzNUwwLjcyNjU0OSAyLjk5OTU5TDAgMy42MzA0NUwyLjg2MDE1IDYuMTMxNTdMOCAwLjYzMDg3Mkw3LjI3ODU3IDBMMi44NjAxNSA0Ljg2NTM1WiIvPgo8L3N2Zz4K);
  --jp-icon-undo: url(data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIxNiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTEyLjUgOGMtMi42NSAwLTUuMDUuOTktNi45IDIuNkwyIDd2OWg5bC0zLjYyLTMuNjJjMS4zOS0xLjE2IDMuMTYtMS44OCA1LjEyLTEuODggMy41NCAwIDYuNTUgMi4zMSA3LjYgNS41bDIuMzctLjc4QzIxLjA4IDExLjAzIDE3LjE1IDggMTIuNSA4eiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-user: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTYiIHZpZXdCb3g9IjAgMCAyNCAyNCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTE2IDdhNCA0IDAgMTEtOCAwIDQgNCAwIDAxOCAwek0xMiAxNGE3IDcgMCAwMC03IDdoMTRhNyA3IDAgMDAtNy03eiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-users: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjQiIGhlaWdodD0iMjQiIHZlcnNpb249IjEuMSIgdmlld0JveD0iMCAwIDM2IDI0IiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPgogPGcgY2xhc3M9ImpwLWljb24zIiB0cmFuc2Zvcm09Im1hdHJpeCgxLjczMjcgMCAwIDEuNzMyNyAtMy42MjgyIC4wOTk1NzcpIiBmaWxsPSIjNjE2MTYxIj4KICA8cGF0aCB0cmFuc2Zvcm09Im1hdHJpeCgxLjUsMCwwLDEuNSwwLC02KSIgZD0ibTEyLjE4NiA3LjUwOThjLTEuMDUzNSAwLTEuOTc1NyAwLjU2NjUtMi40Nzg1IDEuNDEwMiAwLjc1MDYxIDAuMzEyNzcgMS4zOTc0IDAuODI2NDggMS44NzMgMS40NzI3aDMuNDg2M2MwLTEuNTkyLTEuMjg4OS0yLjg4MjgtMi44ODA5LTIuODgyOHoiLz4KICA8cGF0aCBkPSJtMjAuNDY1IDIuMzg5NWEyLjE4ODUgMi4xODg1IDAgMCAxLTIuMTg4NCAyLjE4ODUgMi4xODg1IDIuMTg4NSAwIDAgMS0yLjE4ODUtMi4xODg1IDIuMTg4NSAyLjE4ODUgMCAwIDEgMi4xODg1LTIuMTg4NSAyLjE4ODUgMi4xODg1IDAgMCAxIDIuMTg4NCAyLjE4ODV6Ii8+CiAgPHBhdGggdHJhbnNmb3JtPSJtYXRyaXgoMS41LDAsMCwxLjUsMCwtNikiIGQ9Im0zLjU4OTggOC40MjE5Yy0xLjExMjYgMC0yLjAxMzcgMC45MDExMS0yLjAxMzcgMi4wMTM3aDIuODE0NWMwLjI2Nzk3LTAuMzczMDkgMC41OTA3LTAuNzA0MzUgMC45NTg5OC0wLjk3ODUyLTAuMzQ0MzMtMC42MTY4OC0xLjAwMzEtMS4wMzUyLTEuNzU5OC0xLjAzNTJ6Ii8+CiAgPHBhdGggZD0ibTYuOTE1NCA0LjYyM2ExLjUyOTQgMS41Mjk0IDAgMCAxLTEuNTI5NCAxLjUyOTQgMS41Mjk0IDEuNTI5NCAwIDAgMS0xLjUyOTQtMS41Mjk0IDEuNTI5NCAxLjUyOTQgMCAwIDEgMS41Mjk0LTEuNTI5NCAxLjUyOTQgMS41Mjk0IDAgMCAxIDEuNTI5NCAxLjUyOTR6Ii8+CiAgPHBhdGggZD0ibTYuMTM1IDEzLjUzNWMwLTMuMjM5MiAyLjYyNTktNS44NjUgNS44NjUtNS44NjUgMy4yMzkyIDAgNS44NjUgMi42MjU5IDUuODY1IDUuODY1eiIvPgogIDxjaXJjbGUgY3g9IjEyIiBjeT0iMy43Njg1IiByPSIyLjk2ODUiLz4KIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-vega: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIyIDIyIj4KICA8ZyBjbGFzcz0ianAtaWNvbjEganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjMjEyMTIxIj4KICAgIDxwYXRoIGQ9Ik0xMC42IDUuNGwyLjItMy4ySDIuMnY3LjNsNC02LjZ6Ii8+CiAgICA8cGF0aCBkPSJNMTUuOCAyLjJsLTQuNCA2LjZMNyA2LjNsLTQuOCA4djUuNWgxNy42VjIuMmgtNHptLTcgMTUuNEg1LjV2LTQuNGgzLjN2NC40em00LjQgMEg5LjhWOS44aDMuNHY3Ljh6bTQuNCAwaC0zLjRWNi41aDMuNHYxMS4xeiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-word: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIwIDIwIj4KIDxnIGNsYXNzPSJqcC1pY29uMiIgZmlsbD0iIzQxNDE0MSI+CiAgPHJlY3QgeD0iMiIgeT0iMiIgd2lkdGg9IjE2IiBoZWlnaHQ9IjE2Ii8+CiA8L2c+CiA8ZyBjbGFzcz0ianAtaWNvbi1hY2NlbnQyIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSguNDMgLjA0MDEpIiBmaWxsPSIjZmZmIj4KICA8cGF0aCBkPSJtNC4xNCA4Ljc2cTAuMDY4Mi0xLjg5IDIuNDItMS44OSAxLjE2IDAgMS42OCAwLjQyIDAuNTY3IDAuNDEgMC41NjcgMS4xNnYzLjQ3cTAgMC40NjIgMC41MTQgMC40NjIgMC4xMDMgMCAwLjItMC4wMjMxdjAuNzE0cS0wLjM5OSAwLjEwMy0wLjY1MSAwLjEwMy0wLjQ1MiAwLTAuNjkzLTAuMjItMC4yMzEtMC4yLTAuMjg0LTAuNjYyLTAuOTU2IDAuODcyLTIgMC44NzItMC45MDMgMC0xLjQ3LTAuNDcyLTAuNTI1LTAuNDcyLTAuNTI1LTEuMjYgMC0wLjI2MiAwLjA0NTItMC40NzIgMC4wNTY3LTAuMjIgMC4xMTYtMC4zNzggMC4wNjgyLTAuMTY4IDAuMjMxLTAuMzA0IDAuMTU4LTAuMTQ3IDAuMjYyLTAuMjQyIDAuMTE2LTAuMDkxNCAwLjM2OC0wLjE2OCAwLjI2Mi0wLjA5MTQgMC4zOTktMC4xMjYgMC4xMzYtMC4wNDUyIDAuNDcyLTAuMTAzIDAuMzM2LTAuMDU3OCAwLjUwNC0wLjA3OTggMC4xNTgtMC4wMjMxIDAuNTY3LTAuMDc5OCAwLjU1Ni0wLjA2ODIgMC43NzctMC4yMjEgMC4yMi0wLjE1MiAwLjIyLTAuNDQxdi0wLjI1MnEwLTAuNDMtMC4zNTctMC42NjItMC4zMzYtMC4yMzEtMC45NzYtMC4yMzEtMC42NjIgMC0wLjk5OCAwLjI2Mi0wLjMzNiAwLjI1Mi0wLjM5OSAwLjc5OHptMS44OSAzLjY4cTAuNzg4IDAgMS4yNi0wLjQxIDAuNTA0LTAuNDIgMC41MDQtMC45MDN2LTEuMDVxLTAuMjg0IDAuMTM2LTAuODYxIDAuMjMxLTAuNTY3IDAuMDkxNC0wLjk4NyAwLjE1OC0wLjQyIDAuMDY4Mi0wLjc2NiAwLjMyNi0wLjMzNiAwLjI1Mi0wLjMzNiAwLjcwNHQwLjMwNCAwLjcwNCAwLjg2MSAwLjI1MnoiIHN0cm9rZS13aWR0aD0iMS4wNSIvPgogIDxwYXRoIGQ9Im0xMCA0LjU2aDAuOTQ1djMuMTVxMC42NTEtMC45NzYgMS44OS0wLjk3NiAxLjE2IDAgMS44OSAwLjg0IDAuNjgyIDAuODQgMC42ODIgMi4zMSAwIDEuNDctMC43MDQgMi40Mi0wLjcwNCAwLjg4Mi0xLjg5IDAuODgyLTEuMjYgMC0xLjg5LTEuMDJ2MC43NjZoLTAuODV6bTIuNjIgMy4wNHEtMC43NDYgMC0xLjE2IDAuNjQtMC40NTIgMC42My0wLjQ1MiAxLjY4IDAgMS4wNSAwLjQ1MiAxLjY4dDEuMTYgMC42M3EwLjc3NyAwIDEuMjYtMC42MyAwLjQ5NC0wLjY0IDAuNDk0LTEuNjggMC0xLjA1LTAuNDcyLTEuNjgtMC40NjItMC42NC0xLjI2LTAuNjR6IiBzdHJva2Utd2lkdGg9IjEuMDUiLz4KICA8cGF0aCBkPSJtMi43MyAxNS44IDEzLjYgMC4wMDgxYzAuMDA2OSAwIDAtMi42IDAtMi42IDAtMC4wMDc4LTEuMTUgMC0xLjE1IDAtMC4wMDY5IDAtMC4wMDgzIDEuNS0wLjAwODMgMS41LTJlLTMgLTAuMDAxNC0xMS4zLTAuMDAxNC0xMS4zLTAuMDAxNGwtMC4wMDU5Mi0xLjVjMC0wLjAwNzgtMS4xNyAwLjAwMTMtMS4xNyAwLjAwMTN6IiBzdHJva2Utd2lkdGg9Ii45NzUiLz4KIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-yaml: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIyIDIyIj4KICA8ZyBjbGFzcz0ianAtaWNvbi1jb250cmFzdDIganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjRDgxQjYwIj4KICAgIDxwYXRoIGQ9Ik03LjIgMTguNnYtNS40TDMgNS42aDMuM2wxLjQgMy4xYy4zLjkuNiAxLjYgMSAyLjUuMy0uOC42LTEuNiAxLTIuNWwxLjQtMy4xaDMuNGwtNC40IDcuNnY1LjVsLTIuOS0uMXoiLz4KICAgIDxjaXJjbGUgY2xhc3M9InN0MCIgY3g9IjE3LjYiIGN5PSIxNi41IiByPSIyLjEiLz4KICAgIDxjaXJjbGUgY2xhc3M9InN0MCIgY3g9IjE3LjYiIGN5PSIxMSIgcj0iMi4xIi8+CiAgPC9nPgo8L3N2Zz4K);
}

/* Icon CSS class declarations */

.jp-AddAboveIcon {
  background-image: var(--jp-icon-add-above);
}

.jp-AddBelowIcon {
  background-image: var(--jp-icon-add-below);
}

.jp-AddIcon {
  background-image: var(--jp-icon-add);
}

.jp-BellIcon {
  background-image: var(--jp-icon-bell);
}

.jp-BugDotIcon {
  background-image: var(--jp-icon-bug-dot);
}

.jp-BugIcon {
  background-image: var(--jp-icon-bug);
}

.jp-BuildIcon {
  background-image: var(--jp-icon-build);
}

.jp-CaretDownEmptyIcon {
  background-image: var(--jp-icon-caret-down-empty);
}

.jp-CaretDownEmptyThinIcon {
  background-image: var(--jp-icon-caret-down-empty-thin);
}

.jp-CaretDownIcon {
  background-image: var(--jp-icon-caret-down);
}

.jp-CaretLeftIcon {
  background-image: var(--jp-icon-caret-left);
}

.jp-CaretRightIcon {
  background-image: var(--jp-icon-caret-right);
}

.jp-CaretUpEmptyThinIcon {
  background-image: var(--jp-icon-caret-up-empty-thin);
}

.jp-CaretUpIcon {
  background-image: var(--jp-icon-caret-up);
}

.jp-CaseSensitiveIcon {
  background-image: var(--jp-icon-case-sensitive);
}

.jp-CheckIcon {
  background-image: var(--jp-icon-check);
}

.jp-CircleEmptyIcon {
  background-image: var(--jp-icon-circle-empty);
}

.jp-CircleIcon {
  background-image: var(--jp-icon-circle);
}

.jp-ClearIcon {
  background-image: var(--jp-icon-clear);
}

.jp-CloseIcon {
  background-image: var(--jp-icon-close);
}

.jp-CodeCheckIcon {
  background-image: var(--jp-icon-code-check);
}

.jp-CodeIcon {
  background-image: var(--jp-icon-code);
}

.jp-CollapseAllIcon {
  background-image: var(--jp-icon-collapse-all);
}

.jp-ConsoleIcon {
  background-image: var(--jp-icon-console);
}

.jp-CopyIcon {
  background-image: var(--jp-icon-copy);
}

.jp-CopyrightIcon {
  background-image: var(--jp-icon-copyright);
}

.jp-CutIcon {
  background-image: var(--jp-icon-cut);
}

.jp-DeleteIcon {
  background-image: var(--jp-icon-delete);
}

.jp-DownloadIcon {
  background-image: var(--jp-icon-download);
}

.jp-DuplicateIcon {
  background-image: var(--jp-icon-duplicate);
}

.jp-EditIcon {
  background-image: var(--jp-icon-edit);
}

.jp-EllipsesIcon {
  background-image: var(--jp-icon-ellipses);
}

.jp-ErrorIcon {
  background-image: var(--jp-icon-error);
}

.jp-ExpandAllIcon {
  background-image: var(--jp-icon-expand-all);
}

.jp-ExtensionIcon {
  background-image: var(--jp-icon-extension);
}

.jp-FastForwardIcon {
  background-image: var(--jp-icon-fast-forward);
}

.jp-FileIcon {
  background-image: var(--jp-icon-file);
}

.jp-FileUploadIcon {
  background-image: var(--jp-icon-file-upload);
}

.jp-FilterDotIcon {
  background-image: var(--jp-icon-filter-dot);
}

.jp-FilterIcon {
  background-image: var(--jp-icon-filter);
}

.jp-FilterListIcon {
  background-image: var(--jp-icon-filter-list);
}

.jp-FolderFavoriteIcon {
  background-image: var(--jp-icon-folder-favorite);
}

.jp-FolderIcon {
  background-image: var(--jp-icon-folder);
}

.jp-HomeIcon {
  background-image: var(--jp-icon-home);
}

.jp-Html5Icon {
  background-image: var(--jp-icon-html5);
}

.jp-ImageIcon {
  background-image: var(--jp-icon-image);
}

.jp-InfoIcon {
  background-image: var(--jp-icon-info);
}

.jp-InspectorIcon {
  background-image: var(--jp-icon-inspector);
}

.jp-JsonIcon {
  background-image: var(--jp-icon-json);
}

.jp-JuliaIcon {
  background-image: var(--jp-icon-julia);
}

.jp-JupyterFaviconIcon {
  background-image: var(--jp-icon-jupyter-favicon);
}

.jp-JupyterIcon {
  background-image: var(--jp-icon-jupyter);
}

.jp-JupyterlabWordmarkIcon {
  background-image: var(--jp-icon-jupyterlab-wordmark);
}

.jp-KernelIcon {
  background-image: var(--jp-icon-kernel);
}

.jp-KeyboardIcon {
  background-image: var(--jp-icon-keyboard);
}

.jp-LaunchIcon {
  background-image: var(--jp-icon-launch);
}

.jp-LauncherIcon {
  background-image: var(--jp-icon-launcher);
}

.jp-LineFormIcon {
  background-image: var(--jp-icon-line-form);
}

.jp-LinkIcon {
  background-image: var(--jp-icon-link);
}

.jp-ListIcon {
  background-image: var(--jp-icon-list);
}

.jp-MarkdownIcon {
  background-image: var(--jp-icon-markdown);
}

.jp-MoveDownIcon {
  background-image: var(--jp-icon-move-down);
}

.jp-MoveUpIcon {
  background-image: var(--jp-icon-move-up);
}

.jp-NewFolderIcon {
  background-image: var(--jp-icon-new-folder);
}

.jp-NotTrustedIcon {
  background-image: var(--jp-icon-not-trusted);
}

.jp-NotebookIcon {
  background-image: var(--jp-icon-notebook);
}

.jp-NumberingIcon {
  background-image: var(--jp-icon-numbering);
}

.jp-OfflineBoltIcon {
  background-image: var(--jp-icon-offline-bolt);
}

.jp-PaletteIcon {
  background-image: var(--jp-icon-palette);
}

.jp-PasteIcon {
  background-image: var(--jp-icon-paste);
}

.jp-PdfIcon {
  background-image: var(--jp-icon-pdf);
}

.jp-PythonIcon {
  background-image: var(--jp-icon-python);
}

.jp-RKernelIcon {
  background-image: var(--jp-icon-r-kernel);
}

.jp-ReactIcon {
  background-image: var(--jp-icon-react);
}

.jp-RedoIcon {
  background-image: var(--jp-icon-redo);
}

.jp-RefreshIcon {
  background-image: var(--jp-icon-refresh);
}

.jp-RegexIcon {
  background-image: var(--jp-icon-regex);
}

.jp-RunIcon {
  background-image: var(--jp-icon-run);
}

.jp-RunningIcon {
  background-image: var(--jp-icon-running);
}

.jp-SaveIcon {
  background-image: var(--jp-icon-save);
}

.jp-SearchIcon {
  background-image: var(--jp-icon-search);
}

.jp-SettingsIcon {
  background-image: var(--jp-icon-settings);
}

.jp-ShareIcon {
  background-image: var(--jp-icon-share);
}

.jp-SpreadsheetIcon {
  background-image: var(--jp-icon-spreadsheet);
}

.jp-StopIcon {
  background-image: var(--jp-icon-stop);
}

.jp-TabIcon {
  background-image: var(--jp-icon-tab);
}

.jp-TableRowsIcon {
  background-image: var(--jp-icon-table-rows);
}

.jp-TagIcon {
  background-image: var(--jp-icon-tag);
}

.jp-TerminalIcon {
  background-image: var(--jp-icon-terminal);
}

.jp-TextEditorIcon {
  background-image: var(--jp-icon-text-editor);
}

.jp-TocIcon {
  background-image: var(--jp-icon-toc);
}

.jp-TreeViewIcon {
  background-image: var(--jp-icon-tree-view);
}

.jp-TrustedIcon {
  background-image: var(--jp-icon-trusted);
}

.jp-UndoIcon {
  background-image: var(--jp-icon-undo);
}

.jp-UserIcon {
  background-image: var(--jp-icon-user);
}

.jp-UsersIcon {
  background-image: var(--jp-icon-users);
}

.jp-VegaIcon {
  background-image: var(--jp-icon-vega);
}

.jp-WordIcon {
  background-image: var(--jp-icon-word);
}

.jp-YamlIcon {
  background-image: var(--jp-icon-yaml);
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/**
 * (DEPRECATED) Support for consuming icons as CSS background images
 */

.jp-Icon,
.jp-MaterialIcon {
  background-position: center;
  background-repeat: no-repeat;
  background-size: 16px;
  min-width: 16px;
  min-height: 16px;
}

.jp-Icon-cover {
  background-position: center;
  background-repeat: no-repeat;
  background-size: cover;
}

/**
 * (DEPRECATED) Support for specific CSS icon sizes
 */

.jp-Icon-16 {
  background-size: 16px;
  min-width: 16px;
  min-height: 16px;
}

.jp-Icon-18 {
  background-size: 18px;
  min-width: 18px;
  min-height: 18px;
}

.jp-Icon-20 {
  background-size: 20px;
  min-width: 20px;
  min-height: 20px;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.lm-TabBar .lm-TabBar-addButton {
  align-items: center;
  display: flex;
  padding: 4px;
  padding-bottom: 5px;
  margin-right: 1px;
  background-color: var(--jp-layout-color2);
}

.lm-TabBar .lm-TabBar-addButton:hover {
  background-color: var(--jp-layout-color1);
}

.lm-DockPanel-tabBar .lm-TabBar-tab {
  width: var(--jp-private-horizontal-tab-width);
}

.lm-DockPanel-tabBar .lm-TabBar-content {
  flex: unset;
}

.lm-DockPanel-tabBar[data-orientation='horizontal'] {
  flex: 1 1 auto;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/**
 * Support for icons as inline SVG HTMLElements
 */

/* recolor the primary elements of an icon */
.jp-icon0[fill] {
  fill: var(--jp-inverse-layout-color0);
}

.jp-icon1[fill] {
  fill: var(--jp-inverse-layout-color1);
}

.jp-icon2[fill] {
  fill: var(--jp-inverse-layout-color2);
}

.jp-icon3[fill] {
  fill: var(--jp-inverse-layout-color3);
}

.jp-icon4[fill] {
  fill: var(--jp-inverse-layout-color4);
}

.jp-icon0[stroke] {
  stroke: var(--jp-inverse-layout-color0);
}

.jp-icon1[stroke] {
  stroke: var(--jp-inverse-layout-color1);
}

.jp-icon2[stroke] {
  stroke: var(--jp-inverse-layout-color2);
}

.jp-icon3[stroke] {
  stroke: var(--jp-inverse-layout-color3);
}

.jp-icon4[stroke] {
  stroke: var(--jp-inverse-layout-color4);
}

/* recolor the accent elements of an icon */
.jp-icon-accent0[fill] {
  fill: var(--jp-layout-color0);
}

.jp-icon-accent1[fill] {
  fill: var(--jp-layout-color1);
}

.jp-icon-accent2[fill] {
  fill: var(--jp-layout-color2);
}

.jp-icon-accent3[fill] {
  fill: var(--jp-layout-color3);
}

.jp-icon-accent4[fill] {
  fill: var(--jp-layout-color4);
}

.jp-icon-accent0[stroke] {
  stroke: var(--jp-layout-color0);
}

.jp-icon-accent1[stroke] {
  stroke: var(--jp-layout-color1);
}

.jp-icon-accent2[stroke] {
  stroke: var(--jp-layout-color2);
}

.jp-icon-accent3[stroke] {
  stroke: var(--jp-layout-color3);
}

.jp-icon-accent4[stroke] {
  stroke: var(--jp-layout-color4);
}

/* set the color of an icon to transparent */
.jp-icon-none[fill] {
  fill: none;
}

.jp-icon-none[stroke] {
  stroke: none;
}

/* brand icon colors. Same for light and dark */
.jp-icon-brand0[fill] {
  fill: var(--jp-brand-color0);
}

.jp-icon-brand1[fill] {
  fill: var(--jp-brand-color1);
}

.jp-icon-brand2[fill] {
  fill: var(--jp-brand-color2);
}

.jp-icon-brand3[fill] {
  fill: var(--jp-brand-color3);
}

.jp-icon-brand4[fill] {
  fill: var(--jp-brand-color4);
}

.jp-icon-brand0[stroke] {
  stroke: var(--jp-brand-color0);
}

.jp-icon-brand1[stroke] {
  stroke: var(--jp-brand-color1);
}

.jp-icon-brand2[stroke] {
  stroke: var(--jp-brand-color2);
}

.jp-icon-brand3[stroke] {
  stroke: var(--jp-brand-color3);
}

.jp-icon-brand4[stroke] {
  stroke: var(--jp-brand-color4);
}

/* warn icon colors. Same for light and dark */
.jp-icon-warn0[fill] {
  fill: var(--jp-warn-color0);
}

.jp-icon-warn1[fill] {
  fill: var(--jp-warn-color1);
}

.jp-icon-warn2[fill] {
  fill: var(--jp-warn-color2);
}

.jp-icon-warn3[fill] {
  fill: var(--jp-warn-color3);
}

.jp-icon-warn0[stroke] {
  stroke: var(--jp-warn-color0);
}

.jp-icon-warn1[stroke] {
  stroke: var(--jp-warn-color1);
}

.jp-icon-warn2[stroke] {
  stroke: var(--jp-warn-color2);
}

.jp-icon-warn3[stroke] {
  stroke: var(--jp-warn-color3);
}

/* icon colors that contrast well with each other and most backgrounds */
.jp-icon-contrast0[fill] {
  fill: var(--jp-icon-contrast-color0);
}

.jp-icon-contrast1[fill] {
  fill: var(--jp-icon-contrast-color1);
}

.jp-icon-contrast2[fill] {
  fill: var(--jp-icon-contrast-color2);
}

.jp-icon-contrast3[fill] {
  fill: var(--jp-icon-contrast-color3);
}

.jp-icon-contrast0[stroke] {
  stroke: var(--jp-icon-contrast-color0);
}

.jp-icon-contrast1[stroke] {
  stroke: var(--jp-icon-contrast-color1);
}

.jp-icon-contrast2[stroke] {
  stroke: var(--jp-icon-contrast-color2);
}

.jp-icon-contrast3[stroke] {
  stroke: var(--jp-icon-contrast-color3);
}

.jp-icon-dot[fill] {
  fill: var(--jp-warn-color0);
}

.jp-jupyter-icon-color[fill] {
  fill: var(--jp-jupyter-icon-color, var(--jp-warn-color0));
}

.jp-notebook-icon-color[fill] {
  fill: var(--jp-notebook-icon-color, var(--jp-warn-color0));
}

.jp-json-icon-color[fill] {
  fill: var(--jp-json-icon-color, var(--jp-warn-color1));
}

.jp-console-icon-color[fill] {
  fill: var(--jp-console-icon-color, white);
}

.jp-console-icon-background-color[fill] {
  fill: var(--jp-console-icon-background-color, var(--jp-brand-color1));
}

.jp-terminal-icon-color[fill] {
  fill: var(--jp-terminal-icon-color, var(--jp-layout-color2));
}

.jp-terminal-icon-background-color[fill] {
  fill: var(
    --jp-terminal-icon-background-color,
    var(--jp-inverse-layout-color2)
  );
}

.jp-text-editor-icon-color[fill] {
  fill: var(--jp-text-editor-icon-color, var(--jp-inverse-layout-color3));
}

.jp-inspector-icon-color[fill] {
  fill: var(--jp-inspector-icon-color, var(--jp-inverse-layout-color3));
}

/* CSS for icons in selected filebrowser listing items */
.jp-DirListing-item.jp-mod-selected .jp-icon-selectable[fill] {
  fill: #fff;
}

.jp-DirListing-item.jp-mod-selected .jp-icon-selectable-inverse[fill] {
  fill: var(--jp-brand-color1);
}

/* stylelint-disable selector-max-class, selector-max-compound-selectors */

/**
* TODO: come up with non css-hack solution for showing the busy icon on top
*  of the close icon
* CSS for complex behavior of close icon of tabs in the main area tabbar
*/
.lm-DockPanel-tabBar
  .lm-TabBar-tab.lm-mod-closable.jp-mod-dirty
  > .lm-TabBar-tabCloseIcon
  > :not(:hover)
  > .jp-icon3[fill] {
  fill: none;
}

.lm-DockPanel-tabBar
  .lm-TabBar-tab.lm-mod-closable.jp-mod-dirty
  > .lm-TabBar-tabCloseIcon
  > :not(:hover)
  > .jp-icon-busy[fill] {
  fill: var(--jp-inverse-layout-color3);
}

/* stylelint-enable selector-max-class, selector-max-compound-selectors */

/* CSS for icons in status bar */
#jp-main-statusbar .jp-mod-selected .jp-icon-selectable[fill] {
  fill: #fff;
}

#jp-main-statusbar .jp-mod-selected .jp-icon-selectable-inverse[fill] {
  fill: var(--jp-brand-color1);
}

/* special handling for splash icon CSS. While the theme CSS reloads during
   splash, the splash icon can loose theming. To prevent that, we set a
   default for its color variable */
:root {
  --jp-warn-color0: var(--md-orange-700);
}

/* not sure what to do with this one, used in filebrowser listing */
.jp-DragIcon {
  margin-right: 4px;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/**
 * Support for alt colors for icons as inline SVG HTMLElements
 */

/* alt recolor the primary elements of an icon */
.jp-icon-alt .jp-icon0[fill] {
  fill: var(--jp-layout-color0);
}

.jp-icon-alt .jp-icon1[fill] {
  fill: var(--jp-layout-color1);
}

.jp-icon-alt .jp-icon2[fill] {
  fill: var(--jp-layout-color2);
}

.jp-icon-alt .jp-icon3[fill] {
  fill: var(--jp-layout-color3);
}

.jp-icon-alt .jp-icon4[fill] {
  fill: var(--jp-layout-color4);
}

.jp-icon-alt .jp-icon0[stroke] {
  stroke: var(--jp-layout-color0);
}

.jp-icon-alt .jp-icon1[stroke] {
  stroke: var(--jp-layout-color1);
}

.jp-icon-alt .jp-icon2[stroke] {
  stroke: var(--jp-layout-color2);
}

.jp-icon-alt .jp-icon3[stroke] {
  stroke: var(--jp-layout-color3);
}

.jp-icon-alt .jp-icon4[stroke] {
  stroke: var(--jp-layout-color4);
}

/* alt recolor the accent elements of an icon */
.jp-icon-alt .jp-icon-accent0[fill] {
  fill: var(--jp-inverse-layout-color0);
}

.jp-icon-alt .jp-icon-accent1[fill] {
  fill: var(--jp-inverse-layout-color1);
}

.jp-icon-alt .jp-icon-accent2[fill] {
  fill: var(--jp-inverse-layout-color2);
}

.jp-icon-alt .jp-icon-accent3[fill] {
  fill: var(--jp-inverse-layout-color3);
}

.jp-icon-alt .jp-icon-accent4[fill] {
  fill: var(--jp-inverse-layout-color4);
}

.jp-icon-alt .jp-icon-accent0[stroke] {
  stroke: var(--jp-inverse-layout-color0);
}

.jp-icon-alt .jp-icon-accent1[stroke] {
  stroke: var(--jp-inverse-layout-color1);
}

.jp-icon-alt .jp-icon-accent2[stroke] {
  stroke: var(--jp-inverse-layout-color2);
}

.jp-icon-alt .jp-icon-accent3[stroke] {
  stroke: var(--jp-inverse-layout-color3);
}

.jp-icon-alt .jp-icon-accent4[stroke] {
  stroke: var(--jp-inverse-layout-color4);
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-icon-hoverShow:not(:hover) .jp-icon-hoverShow-content {
  display: none !important;
}

/**
 * Support for hover colors for icons as inline SVG HTMLElements
 */

/**
 * regular colors
 */

/* recolor the primary elements of an icon */
.jp-icon-hover :hover .jp-icon0-hover[fill] {
  fill: var(--jp-inverse-layout-color0);
}

.jp-icon-hover :hover .jp-icon1-hover[fill] {
  fill: var(--jp-inverse-layout-color1);
}

.jp-icon-hover :hover .jp-icon2-hover[fill] {
  fill: var(--jp-inverse-layout-color2);
}

.jp-icon-hover :hover .jp-icon3-hover[fill] {
  fill: var(--jp-inverse-layout-color3);
}

.jp-icon-hover :hover .jp-icon4-hover[fill] {
  fill: var(--jp-inverse-layout-color4);
}

.jp-icon-hover :hover .jp-icon0-hover[stroke] {
  stroke: var(--jp-inverse-layout-color0);
}

.jp-icon-hover :hover .jp-icon1-hover[stroke] {
  stroke: var(--jp-inverse-layout-color1);
}

.jp-icon-hover :hover .jp-icon2-hover[stroke] {
  stroke: var(--jp-inverse-layout-color2);
}

.jp-icon-hover :hover .jp-icon3-hover[stroke] {
  stroke: var(--jp-inverse-layout-color3);
}

.jp-icon-hover :hover .jp-icon4-hover[stroke] {
  stroke: var(--jp-inverse-layout-color4);
}

/* recolor the accent elements of an icon */
.jp-icon-hover :hover .jp-icon-accent0-hover[fill] {
  fill: var(--jp-layout-color0);
}

.jp-icon-hover :hover .jp-icon-accent1-hover[fill] {
  fill: var(--jp-layout-color1);
}

.jp-icon-hover :hover .jp-icon-accent2-hover[fill] {
  fill: var(--jp-layout-color2);
}

.jp-icon-hover :hover .jp-icon-accent3-hover[fill] {
  fill: var(--jp-layout-color3);
}

.jp-icon-hover :hover .jp-icon-accent4-hover[fill] {
  fill: var(--jp-layout-color4);
}

.jp-icon-hover :hover .jp-icon-accent0-hover[stroke] {
  stroke: var(--jp-layout-color0);
}

.jp-icon-hover :hover .jp-icon-accent1-hover[stroke] {
  stroke: var(--jp-layout-color1);
}

.jp-icon-hover :hover .jp-icon-accent2-hover[stroke] {
  stroke: var(--jp-layout-color2);
}

.jp-icon-hover :hover .jp-icon-accent3-hover[stroke] {
  stroke: var(--jp-layout-color3);
}

.jp-icon-hover :hover .jp-icon-accent4-hover[stroke] {
  stroke: var(--jp-layout-color4);
}

/* set the color of an icon to transparent */
.jp-icon-hover :hover .jp-icon-none-hover[fill] {
  fill: none;
}

.jp-icon-hover :hover .jp-icon-none-hover[stroke] {
  stroke: none;
}

/**
 * inverse colors
 */

/* inverse recolor the primary elements of an icon */
.jp-icon-hover.jp-icon-alt :hover .jp-icon0-hover[fill] {
  fill: var(--jp-layout-color0);
}

.jp-icon-hover.jp-icon-alt :hover .jp-icon1-hover[fill] {
  fill: var(--jp-layout-color1);
}

.jp-icon-hover.jp-icon-alt :hover .jp-icon2-hover[fill] {
  fill: var(--jp-layout-color2);
}

.jp-icon-hover.jp-icon-alt :hover .jp-icon3-hover[fill] {
  fill: var(--jp-layout-color3);
}

.jp-icon-hover.jp-icon-alt :hover .jp-icon4-hover[fill] {
  fill: var(--jp-layout-color4);
}

.jp-icon-hover.jp-icon-alt :hover .jp-icon0-hover[stroke] {
  stroke: var(--jp-layout-color0);
}

.jp-icon-hover.jp-icon-alt :hover .jp-icon1-hover[stroke] {
  stroke: var(--jp-layout-color1);
}

.jp-icon-hover.jp-icon-alt :hover .jp-icon2-hover[stroke] {
  stroke: var(--jp-layout-color2);
}

.jp-icon-hover.jp-icon-alt :hover .jp-icon3-hover[stroke] {
  stroke: var(--jp-layout-color3);
}

.jp-icon-hover.jp-icon-alt :hover .jp-icon4-hover[stroke] {
  stroke: var(--jp-layout-color4);
}

/* inverse recolor the accent elements of an icon */
.jp-icon-hover.jp-icon-alt :hover .jp-icon-accent0-hover[fill] {
  fill: var(--jp-inverse-layout-color0);
}

.jp-icon-hover.jp-icon-alt :hover .jp-icon-accent1-hover[fill] {
  fill: var(--jp-inverse-layout-color1);
}

.jp-icon-hover.jp-icon-alt :hover .jp-icon-accent2-hover[fill] {
  fill: var(--jp-inverse-layout-color2);
}

.jp-icon-hover.jp-icon-alt :hover .jp-icon-accent3-hover[fill] {
  fill: var(--jp-inverse-layout-color3);
}

.jp-icon-hover.jp-icon-alt :hover .jp-icon-accent4-hover[fill] {
  fill: var(--jp-inverse-layout-color4);
}

.jp-icon-hover.jp-icon-alt :hover .jp-icon-accent0-hover[stroke] {
  stroke: var(--jp-inverse-layout-color0);
}

.jp-icon-hover.jp-icon-alt :hover .jp-icon-accent1-hover[stroke] {
  stroke: var(--jp-inverse-layout-color1);
}

.jp-icon-hover.jp-icon-alt :hover .jp-icon-accent2-hover[stroke] {
  stroke: var(--jp-inverse-layout-color2);
}

.jp-icon-hover.jp-icon-alt :hover .jp-icon-accent3-hover[stroke] {
  stroke: var(--jp-inverse-layout-color3);
}

.jp-icon-hover.jp-icon-alt :hover .jp-icon-accent4-hover[stroke] {
  stroke: var(--jp-inverse-layout-color4);
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-IFrame {
  width: 100%;
  height: 100%;
}

.jp-IFrame > iframe {
  border: none;
}

/*
When drag events occur, `lm-mod-override-cursor` is added to the body.
Because iframes steal all cursor events, the following two rules are necessary
to suppress pointer events while resize drags are occurring. There may be a
better solution to this problem.
*/
body.lm-mod-override-cursor .jp-IFrame {
  position: relative;
}

body.lm-mod-override-cursor .jp-IFrame::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: transparent;
}

/*-----------------------------------------------------------------------------
| Copyright (c) 2014-2016, Jupyter Development Team.
|
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-HoverBox {
  position: fixed;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-FormGroup-content fieldset {
  border: none;
  padding: 0;
  min-width: 0;
  width: 100%;
}

/* stylelint-disable selector-max-type */

.jp-FormGroup-content fieldset .jp-inputFieldWrapper input,
.jp-FormGroup-content fieldset .jp-inputFieldWrapper select,
.jp-FormGroup-content fieldset .jp-inputFieldWrapper textarea {
  font-size: var(--jp-content-font-size2);
  border-color: var(--jp-input-border-color);
  border-style: solid;
  border-radius: var(--jp-border-radius);
  border-width: 1px;
  padding: 6px 8px;
  background: none;
  color: var(--jp-ui-font-color0);
  height: inherit;
}

.jp-FormGroup-content fieldset input[type='checkbox'] {
  position: relative;
  top: 2px;
  margin-left: 0;
}

.jp-FormGroup-content button.jp-mod-styled {
  cursor: pointer;
}

.jp-FormGroup-content .checkbox label {
  cursor: pointer;
  font-size: var(--jp-content-font-size1);
}

.jp-FormGroup-content .jp-root > fieldset > legend {
  display: none;
}

.jp-FormGroup-content .jp-root > fieldset > p {
  display: none;
}

/** copy of `input.jp-mod-styled:focus` style */
.jp-FormGroup-content fieldset input:focus,
.jp-FormGroup-content fieldset select:focus {
  -moz-outline-radius: unset;
  outline: var(--jp-border-width) solid var(--md-blue-500);
  outline-offset: -1px;
  box-shadow: inset 0 0 4px var(--md-blue-300);
}

.jp-FormGroup-content fieldset input:hover:not(:focus),
.jp-FormGroup-content fieldset select:hover:not(:focus) {
  background-color: var(--jp-border-color2);
}

/* stylelint-enable selector-max-type */

.jp-FormGroup-content .checkbox .field-description {
  /* Disable default description field for checkbox:
   because other widgets do not have description fields,
   we add descriptions to each widget on the field level.
  */
  display: none;
}

.jp-FormGroup-content #root__description {
  display: none;
}

.jp-FormGroup-content .jp-modifiedIndicator {
  width: 5px;
  background-color: var(--jp-brand-color2);
  margin-top: 0;
  margin-left: calc(var(--jp-private-settingeditor-modifier-indent) * -1);
  flex-shrink: 0;
}

.jp-FormGroup-content .jp-modifiedIndicator.jp-errorIndicator {
  background-color: var(--jp-error-color0);
  margin-right: 0.5em;
}

/* RJSF ARRAY style */

.jp-arrayFieldWrapper legend {
  font-size: var(--jp-content-font-size2);
  color: var(--jp-ui-font-color0);
  flex-basis: 100%;
  padding: 4px 0;
  font-weight: var(--jp-content-heading-font-weight);
  border-bottom: 1px solid var(--jp-border-color2);
}

.jp-arrayFieldWrapper .field-description {
  padding: 4px 0;
  white-space: pre-wrap;
}

.jp-arrayFieldWrapper .array-item {
  width: 100%;
  border: 1px solid var(--jp-border-color2);
  border-radius: 4px;
  margin: 4px;
}

.jp-ArrayOperations {
  display: flex;
  margin-left: 8px;
}

.jp-ArrayOperationsButton {
  margin: 2px;
}

.jp-ArrayOperationsButton .jp-icon3[fill] {
  fill: var(--jp-ui-font-color0);
}

button.jp-ArrayOperationsButton.jp-mod-styled:disabled {
  cursor: not-allowed;
  opacity: 0.5;
}

/* RJSF form validation error */

.jp-FormGroup-content .validationErrors {
  color: var(--jp-error-color0);
}

/* Hide panel level error as duplicated the field level error */
.jp-FormGroup-content .panel.errors {
  display: none;
}

/* RJSF normal content (settings-editor) */

.jp-FormGroup-contentNormal {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
}

.jp-FormGroup-contentNormal .jp-FormGroup-contentItem {
  margin-left: 7px;
  color: var(--jp-ui-font-color0);
}

.jp-FormGroup-contentNormal .jp-FormGroup-description {
  flex-basis: 100%;
  padding: 4px 7px;
}

.jp-FormGroup-contentNormal .jp-FormGroup-default {
  flex-basis: 100%;
  padding: 4px 7px;
}

.jp-FormGroup-contentNormal .jp-FormGroup-fieldLabel {
  font-size: var(--jp-content-font-size1);
  font-weight: normal;
  min-width: 120px;
}

.jp-FormGroup-contentNormal fieldset:not(:first-child) {
  margin-left: 7px;
}

.jp-FormGroup-contentNormal .field-array-of-string .array-item {
  /* Display `jp-ArrayOperations` buttons side-by-side with content except
    for small screens where flex-wrap will place them one below the other.
  */
  display: flex;
  align-items: center;
  flex-wrap: wrap;
}

.jp-FormGroup-contentNormal .jp-objectFieldWrapper .form-group {
  padding: 2px 8px 2px var(--jp-private-settingeditor-modifier-indent);
  margin-top: 2px;
}

/* RJSF compact content (metadata-form) */

.jp-FormGroup-content.jp-FormGroup-contentCompact {
  width: 100%;
}

.jp-FormGroup-contentCompact .form-group {
  display: flex;
  padding: 0.5em 0.2em 0.5em 0;
}

.jp-FormGroup-contentCompact
  .jp-FormGroup-compactTitle
  .jp-FormGroup-description {
  font-size: var(--jp-ui-font-size1);
  color: var(--jp-ui-font-color2);
}

.jp-FormGroup-contentCompact .jp-FormGroup-fieldLabel {
  padding-bottom: 0.3em;
}

.jp-FormGroup-contentCompact .jp-inputFieldWrapper .form-control {
  width: 100%;
  box-sizing: border-box;
}

.jp-FormGroup-contentCompact .jp-arrayFieldWrapper .jp-FormGroup-compactTitle {
  padding-bottom: 7px;
}

.jp-FormGroup-contentCompact
  .jp-objectFieldWrapper
  .jp-objectFieldWrapper
  .form-group {
  padding: 2px 8px 2px var(--jp-private-settingeditor-modifier-indent);
  margin-top: 2px;
}

.jp-FormGroup-contentCompact ul.error-detail {
  margin-block-start: 0.5em;
  margin-block-end: 0.5em;
  padding-inline-start: 1em;
}

/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

.jp-SidePanel {
  display: flex;
  flex-direction: column;
  min-width: var(--jp-sidebar-min-width);
  overflow-y: auto;
  color: var(--jp-ui-font-color1);
  background: var(--jp-layout-color1);
  font-size: var(--jp-ui-font-size1);
}

.jp-SidePanel-header {
  flex: 0 0 auto;
  display: flex;
  border-bottom: var(--jp-border-width) solid var(--jp-border-color2);
  font-size: var(--jp-ui-font-size0);
  font-weight: 600;
  letter-spacing: 1px;
  margin: 0;
  padding: 2px;
  text-transform: uppercase;
}

.jp-SidePanel-toolbar {
  flex: 0 0 auto;
}

.jp-SidePanel-content {
  flex: 1 1 auto;
}

.jp-SidePanel-toolbar,
.jp-AccordionPanel-toolbar {
  height: var(--jp-private-toolbar-height);
}

.jp-SidePanel-toolbar.jp-Toolbar-micro {
  display: none;
}

.lm-AccordionPanel .jp-AccordionPanel-title {
  box-sizing: border-box;
  line-height: 25px;
  margin: 0;
  display: flex;
  align-items: center;
  background: var(--jp-layout-color1);
  color: var(--jp-ui-font-color1);
  border-bottom: var(--jp-border-width) solid var(--jp-toolbar-border-color);
  box-shadow: var(--jp-toolbar-box-shadow);
  font-size: var(--jp-ui-font-size0);
}

.jp-AccordionPanel-title {
  cursor: pointer;
  user-select: none;
  -moz-user-select: none;
  -webkit-user-select: none;
  text-transform: uppercase;
}

.lm-AccordionPanel[data-orientation='horizontal'] > .jp-AccordionPanel-title {
  /* Title is rotated for horizontal accordion panel using CSS */
  display: block;
  transform-origin: top left;
  transform: rotate(-90deg) translate(-100%);
}

.jp-AccordionPanel-title .lm-AccordionPanel-titleLabel {
  user-select: none;
  text-overflow: ellipsis;
  white-space: nowrap;
  overflow: hidden;
}

.jp-AccordionPanel-title .lm-AccordionPanel-titleCollapser {
  transform: rotate(-90deg);
  margin: auto 0;
  height: 16px;
}

.jp-AccordionPanel-title.lm-mod-expanded .lm-AccordionPanel-titleCollapser {
  transform: rotate(0deg);
}

.lm-AccordionPanel .jp-AccordionPanel-toolbar {
  background: none;
  box-shadow: none;
  border: none;
  margin-left: auto;
}

.lm-AccordionPanel .lm-SplitPanel-handle:hover {
  background: var(--jp-layout-color3);
}

.jp-text-truncated {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

/*-----------------------------------------------------------------------------
| Copyright (c) 2017, Jupyter Development Team.
|
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-Spinner {
  position: absolute;
  display: flex;
  justify-content: center;
  align-items: center;
  z-index: 10;
  left: 0;
  top: 0;
  width: 100%;
  height: 100%;
  background: var(--jp-layout-color0);
  outline: none;
}

.jp-SpinnerContent {
  font-size: 10px;
  margin: 50px auto;
  text-indent: -9999em;
  width: 3em;
  height: 3em;
  border-radius: 50%;
  background: var(--jp-brand-color3);
  background: linear-gradient(
    to right,
    #f37626 10%,
    rgba(255, 255, 255, 0) 42%
  );
  position: relative;
  animation: load3 1s infinite linear, fadeIn 1s;
}

.jp-SpinnerContent::before {
  width: 50%;
  height: 50%;
  background: #f37626;
  border-radius: 100% 0 0;
  position: absolute;
  top: 0;
  left: 0;
  content: '';
}

.jp-SpinnerContent::after {
  background: var(--jp-layout-color0);
  width: 75%;
  height: 75%;
  border-radius: 50%;
  content: '';
  margin: auto;
  position: absolute;
  top: 0;
  left: 0;
  bottom: 0;
  right: 0;
}

@keyframes fadeIn {
  0% {
    opacity: 0;
  }

  100% {
    opacity: 1;
  }
}

@keyframes load3 {
  0% {
    transform: rotate(0deg);
  }

  100% {
    transform: rotate(360deg);
  }
}

/*-----------------------------------------------------------------------------
| Copyright (c) 2014-2017, Jupyter Development Team.
|
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

button.jp-mod-styled {
  font-size: var(--jp-ui-font-size1);
  color: var(--jp-ui-font-color0);
  border: none;
  box-sizing: border-box;
  text-align: center;
  line-height: 32px;
  height: 32px;
  padding: 0 12px;
  letter-spacing: 0.8px;
  outline: none;
  appearance: none;
  -webkit-appearance: none;
  -moz-appearance: none;
}

input.jp-mod-styled {
  background: var(--jp-input-background);
  height: 28px;
  box-sizing: border-box;
  border: var(--jp-border-width) solid var(--jp-border-color1);
  padding-left: 7px;
  padding-right: 7px;
  font-size: var(--jp-ui-font-size2);
  color: var(--jp-ui-font-color0);
  outline: none;
  appearance: none;
  -webkit-appearance: none;
  -moz-appearance: none;
}

input[type='checkbox'].jp-mod-styled {
  appearance: checkbox;
  -webkit-appearance: checkbox;
  -moz-appearance: checkbox;
  height: auto;
}

input.jp-mod-styled:focus {
  border: var(--jp-border-width) solid var(--md-blue-500);
  box-shadow: inset 0 0 4px var(--md-blue-300);
}

.jp-select-wrapper {
  display: flex;
  position: relative;
  flex-direction: column;
  padding: 1px;
  background-color: var(--jp-layout-color1);
  box-sizing: border-box;
  margin-bottom: 12px;
}

.jp-select-wrapper:not(.multiple) {
  height: 28px;
}

.jp-select-wrapper.jp-mod-focused select.jp-mod-styled {
  border: var(--jp-border-width) solid var(--jp-input-active-border-color);
  box-shadow: var(--jp-input-box-shadow);
  background-color: var(--jp-input-active-background);
}

select.jp-mod-styled:hover {
  cursor: pointer;
  color: var(--jp-ui-font-color0);
  background-color: var(--jp-input-hover-background);
  box-shadow: inset 0 0 1px rgba(0, 0, 0, 0.5);
}

select.jp-mod-styled {
  flex: 1 1 auto;
  width: 100%;
  font-size: var(--jp-ui-font-size2);
  background: var(--jp-input-background);
  color: var(--jp-ui-font-color0);
  padding: 0 25px 0 8px;
  border: var(--jp-border-width) solid var(--jp-input-border-color);
  border-radius: 0;
  outline: none;
  appearance: none;
  -webkit-appearance: none;
  -moz-appearance: none;
}

select.jp-mod-styled:not([multiple]) {
  height: 32px;
}

select.jp-mod-styled[multiple] {
  max-height: 200px;
  overflow-y: auto;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-switch {
  display: flex;
  align-items: center;
  padding-left: 4px;
  padding-right: 4px;
  font-size: var(--jp-ui-font-size1);
  background-color: transparent;
  color: var(--jp-ui-font-color1);
  border: none;
  height: 20px;
}

.jp-switch:hover {
  background-color: var(--jp-layout-color2);
}

.jp-switch-label {
  margin-right: 5px;
  font-family: var(--jp-ui-font-family);
}

.jp-switch-track {
  cursor: pointer;
  background-color: var(--jp-switch-color, var(--jp-border-color1));
  -webkit-transition: 0.4s;
  transition: 0.4s;
  border-radius: 34px;
  height: 16px;
  width: 35px;
  position: relative;
}

.jp-switch-track::before {
  content: '';
  position: absolute;
  height: 10px;
  width: 10px;
  margin: 3px;
  left: 0;
  background-color: var(--jp-ui-inverse-font-color1);
  -webkit-transition: 0.4s;
  transition: 0.4s;
  border-radius: 50%;
}

.jp-switch[aria-checked='true'] .jp-switch-track {
  background-color: var(--jp-switch-true-position-color, var(--jp-warn-color0));
}

.jp-switch[aria-checked='true'] .jp-switch-track::before {
  /* track width (35) - margins (3 + 3) - thumb width (10) */
  left: 19px;
}

/*-----------------------------------------------------------------------------
| Copyright (c) 2014-2016, Jupyter Development Team.
|
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

:root {
  --jp-private-toolbar-height: calc(
    28px + var(--jp-border-width)
  ); /* leave 28px for content */
}

.jp-Toolbar {
  color: var(--jp-ui-font-color1);
  flex: 0 0 auto;
  display: flex;
  flex-direction: row;
  border-bottom: var(--jp-border-width) solid var(--jp-toolbar-border-color);
  box-shadow: var(--jp-toolbar-box-shadow);
  background: var(--jp-toolbar-background);
  min-height: var(--jp-toolbar-micro-height);
  padding: 2px;
  z-index: 8;
  overflow-x: hidden;
}

/* Toolbar items */

.jp-Toolbar > .jp-Toolbar-item.jp-Toolbar-spacer {
  flex-grow: 1;
  flex-shrink: 1;
}

.jp-Toolbar-item.jp-Toolbar-kernelStatus {
  display: inline-block;
  width: 32px;
  background-repeat: no-repeat;
  background-position: center;
  background-size: 16px;
}

.jp-Toolbar > .jp-Toolbar-item {
  flex: 0 0 auto;
  display: flex;
  padding-left: 1px;
  padding-right: 1px;
  font-size: var(--jp-ui-font-size1);
  line-height: var(--jp-private-toolbar-height);
  height: 100%;
}

/* Toolbar buttons */

/* This is the div we use to wrap the react component into a Widget */
div.jp-ToolbarButton {
  color: transparent;
  border: none;
  box-sizing: border-box;
  outline: none;
  appearance: none;
  -webkit-appearance: none;
  -moz-appearance: none;
  padding: 0;
  margin: 0;
}

button.jp-ToolbarButtonComponent {
  background: var(--jp-layout-color1);
  border: none;
  box-sizing: border-box;
  outline: none;
  appearance: none;
  -webkit-appearance: none;
  -moz-appearance: none;
  padding: 0 6px;
  margin: 0;
  height: 24px;
  border-radius: var(--jp-border-radius);
  display: flex;
  align-items: center;
  text-align: center;
  font-size: 14px;
  min-width: unset;
  min-height: unset;
}

button.jp-ToolbarButtonComponent:disabled {
  opacity: 0.4;
}

button.jp-ToolbarButtonComponent > span {
  padding: 0;
  flex: 0 0 auto;
}

button.jp-ToolbarButtonComponent .jp-ToolbarButtonComponent-label {
  font-size: var(--jp-ui-font-size1);
  line-height: 100%;
  padding-left: 2px;
  color: var(--jp-ui-font-color1);
  font-family: var(--jp-ui-font-family);
}

#jp-main-dock-panel[data-mode='single-document']
  .jp-MainAreaWidget
  > .jp-Toolbar.jp-Toolbar-micro {
  padding: 0;
  min-height: 0;
}

#jp-main-dock-panel[data-mode='single-document']
  .jp-MainAreaWidget
  > .jp-Toolbar {
  border: none;
  box-shadow: none;
}

/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

.jp-WindowedPanel-outer {
  position: relative;
  overflow-y: auto;
}

.jp-WindowedPanel-inner {
  position: relative;
}

.jp-WindowedPanel-window {
  position: absolute;
  left: 0;
  right: 0;
  overflow: visible;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/* Sibling imports */

body {
  color: var(--jp-ui-font-color1);
  font-size: var(--jp-ui-font-size1);
}

/* Disable native link decoration styles everywhere outside of dialog boxes */
a {
  text-decoration: unset;
  color: unset;
}

a:hover {
  text-decoration: unset;
  color: unset;
}

/* Accessibility for links inside dialog box text */
.jp-Dialog-content a {
  text-decoration: revert;
  color: var(--jp-content-link-color);
}

.jp-Dialog-content a:hover {
  text-decoration: revert;
}

/* Styles for ui-components */
.jp-Button {
  color: var(--jp-ui-font-color2);
  border-radius: var(--jp-border-radius);
  padding: 0 12px;
  font-size: var(--jp-ui-font-size1);

  /* Copy from blueprint 3 */
  display: inline-flex;
  flex-direction: row;
  border: none;
  cursor: pointer;
  align-items: center;
  justify-content: center;
  text-align: left;
  vertical-align: middle;
  min-height: 30px;
  min-width: 30px;
}

.jp-Button:disabled {
  cursor: not-allowed;
}

.jp-Button:empty {
  padding: 0 !important;
}

.jp-Button.jp-mod-small {
  min-height: 24px;
  min-width: 24px;
  font-size: 12px;
  padding: 0 7px;
}

/* Use our own theme for hover styles */
.jp-Button.jp-mod-minimal:hover {
  background-color: var(--jp-layout-color2);
}

.jp-Button.jp-mod-minimal {
  background: none;
}

.jp-InputGroup {
  display: block;
  position: relative;
}

.jp-InputGroup input {
  box-sizing: border-box;
  border: none;
  border-radius: 0;
  background-color: transparent;
  color: var(--jp-ui-font-color0);
  box-shadow: inset 0 0 0 var(--jp-border-width) var(--jp-input-border-color);
  padding-bottom: 0;
  padding-top: 0;
  padding-left: 10px;
  padding-right: 28px;
  position: relative;
  width: 100%;
  -webkit-appearance: none;
  -moz-appearance: none;
  appearance: none;
  font-size: 14px;
  font-weight: 400;
  height: 30px;
  line-height: 30px;
  outline: none;
  vertical-align: middle;
}

.jp-InputGroup input:focus {
  box-shadow: inset 0 0 0 var(--jp-border-width)
      var(--jp-input-active-box-shadow-color),
    inset 0 0 0 3px var(--jp-input-active-box-shadow-color);
}

.jp-InputGroup input:disabled {
  cursor: not-allowed;
  resize: block;
  background-color: var(--jp-layout-color2);
  color: var(--jp-ui-font-color2);
}

.jp-InputGroup input:disabled ~ span {
  cursor: not-allowed;
  color: var(--jp-ui-font-color2);
}

.jp-InputGroup input::placeholder,
input::placeholder {
  color: var(--jp-ui-font-color2);
}

.jp-InputGroupAction {
  position: absolute;
  bottom: 1px;
  right: 0;
  padding: 6px;
}

.jp-HTMLSelect.jp-DefaultStyle select {
  background-color: initial;
  border: none;
  border-radius: 0;
  box-shadow: none;
  color: var(--jp-ui-font-color0);
  display: block;
  font-size: var(--jp-ui-font-size1);
  font-family: var(--jp-ui-font-family);
  height: 24px;
  line-height: 14px;
  padding: 0 25px 0 10px;
  text-align: left;
  -moz-appearance: none;
  -webkit-appearance: none;
}

.jp-HTMLSelect.jp-DefaultStyle select:disabled {
  background-color: var(--jp-layout-color2);
  color: var(--jp-ui-font-color2);
  cursor: not-allowed;
  resize: block;
}

.jp-HTMLSelect.jp-DefaultStyle select:disabled ~ span {
  cursor: not-allowed;
}

/* Use our own theme for hover and option styles */
/* stylelint-disable-next-line selector-max-type */
.jp-HTMLSelect.jp-DefaultStyle select:hover,
.jp-HTMLSelect.jp-DefaultStyle select > option {
  background-color: var(--jp-layout-color2);
  color: var(--jp-ui-font-color0);
}

select {
  box-sizing: border-box;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Styles
|----------------------------------------------------------------------------*/

.jp-StatusBar-Widget {
  display: flex;
  align-items: center;
  background: var(--jp-layout-color2);
  min-height: var(--jp-statusbar-height);
  justify-content: space-between;
  padding: 0 10px;
}

.jp-StatusBar-Left {
  display: flex;
  align-items: center;
  flex-direction: row;
}

.jp-StatusBar-Middle {
  display: flex;
  align-items: center;
}

.jp-StatusBar-Right {
  display: flex;
  align-items: center;
  flex-direction: row-reverse;
}

.jp-StatusBar-Item {
  max-height: var(--jp-statusbar-height);
  margin: 0 2px;
  height: var(--jp-statusbar-height);
  white-space: nowrap;
  text-overflow: ellipsis;
  color: var(--jp-ui-font-color1);
  padding: 0 6px;
}

.jp-mod-highlighted:hover {
  background-color: var(--jp-layout-color3);
}

.jp-mod-clicked {
  background-color: var(--jp-brand-color1);
}

.jp-mod-clicked:hover {
  background-color: var(--jp-brand-color0);
}

.jp-mod-clicked .jp-StatusBar-TextItem {
  color: var(--jp-ui-inverse-font-color1);
}

.jp-StatusBar-HoverItem {
  box-shadow: '0px 4px 4px rgba(0, 0, 0, 0.25)';
}

.jp-StatusBar-TextItem {
  font-size: var(--jp-ui-font-size1);
  font-family: var(--jp-ui-font-family);
  line-height: 24px;
  color: var(--jp-ui-font-color1);
}

.jp-StatusBar-GroupItem {
  display: flex;
  align-items: center;
  flex-direction: row;
}

.jp-Statusbar-ProgressCircle svg {
  display: block;
  margin: 0 auto;
  width: 16px;
  height: 24px;
  align-self: normal;
}

.jp-Statusbar-ProgressCircle path {
  fill: var(--jp-inverse-layout-color3);
}

.jp-Statusbar-ProgressBar-progress-bar {
  height: 10px;
  width: 100px;
  border: solid 0.25px var(--jp-brand-color2);
  border-radius: 3px;
  overflow: hidden;
  align-self: center;
}

.jp-Statusbar-ProgressBar-progress-bar > div {
  background-color: var(--jp-brand-color2);
  background-image: linear-gradient(
    -45deg,
    rgba(255, 255, 255, 0.2) 25%,
    transparent 25%,
    transparent 50%,
    rgba(255, 255, 255, 0.2) 50%,
    rgba(255, 255, 255, 0.2) 75%,
    transparent 75%,
    transparent
  );
  background-size: 40px 40px;
  float: left;
  width: 0%;
  height: 100%;
  font-size: 12px;
  line-height: 14px;
  color: #fff;
  text-align: center;
  animation: jp-Statusbar-ExecutionTime-progress-bar 2s linear infinite;
}

.jp-Statusbar-ProgressBar-progress-bar p {
  color: var(--jp-ui-font-color1);
  font-family: var(--jp-ui-font-family);
  font-size: var(--jp-ui-font-size1);
  line-height: 10px;
  width: 100px;
}

@keyframes jp-Statusbar-ExecutionTime-progress-bar {
  0% {
    background-position: 0 0;
  }

  100% {
    background-position: 40px 40px;
  }
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Variables
|----------------------------------------------------------------------------*/

:root {
  --jp-private-commandpalette-search-height: 28px;
}

/*-----------------------------------------------------------------------------
| Overall styles
|----------------------------------------------------------------------------*/

.lm-CommandPalette {
  padding-bottom: 0;
  color: var(--jp-ui-font-color1);
  background: var(--jp-layout-color1);

  /* This is needed so that all font sizing of children done in ems is
   * relative to this base size */
  font-size: var(--jp-ui-font-size1);
}

/*-----------------------------------------------------------------------------
| Modal variant
|----------------------------------------------------------------------------*/

.jp-ModalCommandPalette {
  position: absolute;
  z-index: 10000;
  top: 38px;
  left: 30%;
  margin: 0;
  padding: 4px;
  width: 40%;
  box-shadow: var(--jp-elevation-z4);
  border-radius: 4px;
  background: var(--jp-layout-color0);
}

.jp-ModalCommandPalette .lm-CommandPalette {
  max-height: 40vh;
}

.jp-ModalCommandPalette .lm-CommandPalette .lm-close-icon::after {
  display: none;
}

.jp-ModalCommandPalette .lm-CommandPalette .lm-CommandPalette-header {
  display: none;
}

.jp-ModalCommandPalette .lm-CommandPalette .lm-CommandPalette-item {
  margin-left: 4px;
  margin-right: 4px;
}

.jp-ModalCommandPalette
  .lm-CommandPalette
  .lm-CommandPalette-item.lm-mod-disabled {
  display: none;
}

/*-----------------------------------------------------------------------------
| Search
|----------------------------------------------------------------------------*/

.lm-CommandPalette-search {
  padding: 4px;
  background-color: var(--jp-layout-color1);
  z-index: 2;
}

.lm-CommandPalette-wrapper {
  overflow: overlay;
  padding: 0 9px;
  background-color: var(--jp-input-active-background);
  height: 30px;
  box-shadow: inset 0 0 0 var(--jp-border-width) var(--jp-input-border-color);
}

.lm-CommandPalette.lm-mod-focused .lm-CommandPalette-wrapper {
  box-shadow: inset 0 0 0 1px var(--jp-input-active-box-shadow-color),
    inset 0 0 0 3px var(--jp-input-active-box-shadow-color);
}

.jp-SearchIconGroup {
  color: white;
  background-color: var(--jp-brand-color1);
  position: absolute;
  top: 4px;
  right: 4px;
  padding: 5px 5px 1px;
}

.jp-SearchIconGroup svg {
  height: 20px;
  width: 20px;
}

.jp-SearchIconGroup .jp-icon3[fill] {
  fill: var(--jp-layout-color0);
}

.lm-CommandPalette-input {
  background: transparent;
  width: calc(100% - 18px);
  float: left;
  border: none;
  outline: none;
  font-size: var(--jp-ui-font-size1);
  color: var(--jp-ui-font-color0);
  line-height: var(--jp-private-commandpalette-search-height);
}

.lm-CommandPalette-input::-webkit-input-placeholder,
.lm-CommandPalette-input::-moz-placeholder,
.lm-CommandPalette-input:-ms-input-placeholder {
  color: var(--jp-ui-font-color2);
  font-size: var(--jp-ui-font-size1);
}

/*-----------------------------------------------------------------------------
| Results
|----------------------------------------------------------------------------*/

.lm-CommandPalette-header:first-child {
  margin-top: 0;
}

.lm-CommandPalette-header {
  border-bottom: solid var(--jp-border-width) var(--jp-border-color2);
  color: var(--jp-ui-font-color1);
  cursor: pointer;
  display: flex;
  font-size: var(--jp-ui-font-size0);
  font-weight: 600;
  letter-spacing: 1px;
  margin-top: 8px;
  padding: 8px 0 8px 12px;
  text-transform: uppercase;
}

.lm-CommandPalette-header.lm-mod-active {
  background: var(--jp-layout-color2);
}

.lm-CommandPalette-header > mark {
  background-color: transparent;
  font-weight: bold;
  color: var(--jp-ui-font-color1);
}

.lm-CommandPalette-item {
  padding: 4px 12px 4px 4px;
  color: var(--jp-ui-font-color1);
  font-size: var(--jp-ui-font-size1);
  font-weight: 400;
  display: flex;
}

.lm-CommandPalette-item.lm-mod-disabled {
  color: var(--jp-ui-font-color2);
}

.lm-CommandPalette-item.lm-mod-active {
  color: var(--jp-ui-inverse-font-color1);
  background: var(--jp-brand-color1);
}

.lm-CommandPalette-item.lm-mod-active .lm-CommandPalette-itemLabel > mark {
  color: var(--jp-ui-inverse-font-color0);
}

.lm-CommandPalette-item.lm-mod-active .jp-icon-selectable[fill] {
  fill: var(--jp-layout-color0);
}

.lm-CommandPalette-item.lm-mod-active:hover:not(.lm-mod-disabled) {
  color: var(--jp-ui-inverse-font-color1);
  background: var(--jp-brand-color1);
}

.lm-CommandPalette-item:hover:not(.lm-mod-active):not(.lm-mod-disabled) {
  background: var(--jp-layout-color2);
}

.lm-CommandPalette-itemContent {
  overflow: hidden;
}

.lm-CommandPalette-itemLabel > mark {
  color: var(--jp-ui-font-color0);
  background-color: transparent;
  font-weight: bold;
}

.lm-CommandPalette-item.lm-mod-disabled mark {
  color: var(--jp-ui-font-color2);
}

.lm-CommandPalette-item .lm-CommandPalette-itemIcon {
  margin: 0 4px 0 0;
  position: relative;
  width: 16px;
  top: 2px;
  flex: 0 0 auto;
}

.lm-CommandPalette-item.lm-mod-disabled .lm-CommandPalette-itemIcon {
  opacity: 0.6;
}

.lm-CommandPalette-item .lm-CommandPalette-itemShortcut {
  flex: 0 0 auto;
}

.lm-CommandPalette-itemCaption {
  display: none;
}

.lm-CommandPalette-content {
  background-color: var(--jp-layout-color1);
}

.lm-CommandPalette-content:empty::after {
  content: 'No results';
  margin: auto;
  margin-top: 20px;
  width: 100px;
  display: block;
  font-size: var(--jp-ui-font-size2);
  font-family: var(--jp-ui-font-family);
  font-weight: lighter;
}

.lm-CommandPalette-emptyMessage {
  text-align: center;
  margin-top: 24px;
  line-height: 1.32;
  padding: 0 8px;
  color: var(--jp-content-font-color3);
}

/*-----------------------------------------------------------------------------
| Copyright (c) 2014-2017, Jupyter Development Team.
|
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-Dialog {
  position: absolute;
  z-index: 10000;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  top: 0;
  left: 0;
  margin: 0;
  padding: 0;
  width: 100%;
  height: 100%;
  background: var(--jp-dialog-background);
}

.jp-Dialog-content {
  display: flex;
  flex-direction: column;
  margin-left: auto;
  margin-right: auto;
  background: var(--jp-layout-color1);
  padding: 24px 24px 12px;
  min-width: 300px;
  min-height: 150px;
  max-width: 1000px;
  max-height: 500px;
  box-sizing: border-box;
  box-shadow: var(--jp-elevation-z20);
  word-wrap: break-word;
  border-radius: var(--jp-border-radius);

  /* This is needed so that all font sizing of children done in ems is
   * relative to this base size */
  font-size: var(--jp-ui-font-size1);
  color: var(--jp-ui-font-color1);
  resize: both;
}

.jp-Dialog-content.jp-Dialog-content-small {
  max-width: 500px;
}

.jp-Dialog-button {
  overflow: visible;
}

button.jp-Dialog-button:focus {
  outline: 1px solid var(--jp-brand-color1);
  outline-offset: 4px;
  -moz-outline-radius: 0;
}

button.jp-Dialog-button:focus::-moz-focus-inner {
  border: 0;
}

button.jp-Dialog-button.jp-mod-styled.jp-mod-accept:focus,
button.jp-Dialog-button.jp-mod-styled.jp-mod-warn:focus,
button.jp-Dialog-button.jp-mod-styled.jp-mod-reject:focus {
  outline-offset: 4px;
  -moz-outline-radius: 0;
}

button.jp-Dialog-button.jp-mod-styled.jp-mod-accept:focus {
  outline: 1px solid var(--jp-accept-color-normal, var(--jp-brand-color1));
}

button.jp-Dialog-button.jp-mod-styled.jp-mod-warn:focus {
  outline: 1px solid var(--jp-warn-color-normal, var(--jp-error-color1));
}

button.jp-Dialog-button.jp-mod-styled.jp-mod-reject:focus {
  outline: 1px solid var(--jp-reject-color-normal, var(--md-grey-600));
}

button.jp-Dialog-close-button {
  padding: 0;
  height: 100%;
  min-width: unset;
  min-height: unset;
}

.jp-Dialog-header {
  display: flex;
  justify-content: space-between;
  flex: 0 0 auto;
  padding-bottom: 12px;
  font-size: var(--jp-ui-font-size3);
  font-weight: 400;
  color: var(--jp-ui-font-color1);
}

.jp-Dialog-body {
  display: flex;
  flex-direction: column;
  flex: 1 1 auto;
  font-size: var(--jp-ui-font-size1);
  background: var(--jp-layout-color1);
  color: var(--jp-ui-font-color1);
  overflow: auto;
}

.jp-Dialog-footer {
  display: flex;
  flex-direction: row;
  justify-content: flex-end;
  align-items: center;
  flex: 0 0 auto;
  margin-left: -12px;
  margin-right: -12px;
  padding: 12px;
}

.jp-Dialog-checkbox {
  padding-right: 5px;
}

.jp-Dialog-checkbox > input:focus-visible {
  outline: 1px solid var(--jp-input-active-border-color);
  outline-offset: 1px;
}

.jp-Dialog-spacer {
  flex: 1 1 auto;
}

.jp-Dialog-title {
  overflow: hidden;
  white-space: nowrap;
  text-overflow: ellipsis;
}

.jp-Dialog-body > .jp-select-wrapper {
  width: 100%;
}

.jp-Dialog-body > button {
  padding: 0 16px;
}

.jp-Dialog-body > label {
  line-height: 1.4;
  color: var(--jp-ui-font-color0);
}

.jp-Dialog-button.jp-mod-styled:not(:last-child) {
  margin-right: 12px;
}

/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

.jp-Input-Boolean-Dialog {
  flex-direction: row-reverse;
  align-items: end;
  width: 100%;
}

.jp-Input-Boolean-Dialog > label {
  flex: 1 1 auto;
}

/*-----------------------------------------------------------------------------
| Copyright (c) 2014-2016, Jupyter Development Team.
|
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-MainAreaWidget > :focus {
  outline: none;
}

.jp-MainAreaWidget .jp-MainAreaWidget-error {
  padding: 6px;
}

.jp-MainAreaWidget .jp-MainAreaWidget-error > pre {
  width: auto;
  padding: 10px;
  background: var(--jp-error-color3);
  border: var(--jp-border-width) solid var(--jp-error-color1);
  border-radius: var(--jp-border-radius);
  color: var(--jp-ui-font-color1);
  font-size: var(--jp-ui-font-size1);
  white-space: pre-wrap;
  word-wrap: break-word;
}

/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

/**
 * google-material-color v1.2.6
 * https://github.com/danlevan/google-material-color
 */
:root {
  --md-red-50: #ffebee;
  --md-red-100: #ffcdd2;
  --md-red-200: #ef9a9a;
  --md-red-300: #e57373;
  --md-red-400: #ef5350;
  --md-red-500: #f44336;
  --md-red-600: #e53935;
  --md-red-700: #d32f2f;
  --md-red-800: #c62828;
  --md-red-900: #b71c1c;
  --md-red-A100: #ff8a80;
  --md-red-A200: #ff5252;
  --md-red-A400: #ff1744;
  --md-red-A700: #d50000;
  --md-pink-50: #fce4ec;
  --md-pink-100: #f8bbd0;
  --md-pink-200: #f48fb1;
  --md-pink-300: #f06292;
  --md-pink-400: #ec407a;
  --md-pink-500: #e91e63;
  --md-pink-600: #d81b60;
  --md-pink-700: #c2185b;
  --md-pink-800: #ad1457;
  --md-pink-900: #880e4f;
  --md-pink-A100: #ff80ab;
  --md-pink-A200: #ff4081;
  --md-pink-A400: #f50057;
  --md-pink-A700: #c51162;
  --md-purple-50: #f3e5f5;
  --md-purple-100: #e1bee7;
  --md-purple-200: #ce93d8;
  --md-purple-300: #ba68c8;
  --md-purple-400: #ab47bc;
  --md-purple-500: #9c27b0;
  --md-purple-600: #8e24aa;
  --md-purple-700: #7b1fa2;
  --md-purple-800: #6a1b9a;
  --md-purple-900: #4a148c;
  --md-purple-A100: #ea80fc;
  --md-purple-A200: #e040fb;
  --md-purple-A400: #d500f9;
  --md-purple-A700: #a0f;
  --md-deep-purple-50: #ede7f6;
  --md-deep-purple-100: #d1c4e9;
  --md-deep-purple-200: #b39ddb;
  --md-deep-purple-300: #9575cd;
  --md-deep-purple-400: #7e57c2;
  --md-deep-purple-500: #673ab7;
  --md-deep-purple-600: #5e35b1;
  --md-deep-purple-700: #512da8;
  --md-deep-purple-800: #4527a0;
  --md-deep-purple-900: #311b92;
  --md-deep-purple-A100: #b388ff;
  --md-deep-purple-A200: #7c4dff;
  --md-deep-purple-A400: #651fff;
  --md-deep-purple-A700: #6200ea;
  --md-indigo-50: #e8eaf6;
  --md-indigo-100: #c5cae9;
  --md-indigo-200: #9fa8da;
  --md-indigo-300: #7986cb;
  --md-indigo-400: #5c6bc0;
  --md-indigo-500: #3f51b5;
  --md-indigo-600: #3949ab;
  --md-indigo-700: #303f9f;
  --md-indigo-800: #283593;
  --md-indigo-900: #1a237e;
  --md-indigo-A100: #8c9eff;
  --md-indigo-A200: #536dfe;
  --md-indigo-A400: #3d5afe;
  --md-indigo-A700: #304ffe;
  --md-blue-50: #e3f2fd;
  --md-blue-100: #bbdefb;
  --md-blue-200: #90caf9;
  --md-blue-300: #64b5f6;
  --md-blue-400: #42a5f5;
  --md-blue-500: #2196f3;
  --md-blue-600: #1e88e5;
  --md-blue-700: #1976d2;
  --md-blue-800: #1565c0;
  --md-blue-900: #0d47a1;
  --md-blue-A100: #82b1ff;
  --md-blue-A200: #448aff;
  --md-blue-A400: #2979ff;
  --md-blue-A700: #2962ff;
  --md-light-blue-50: #e1f5fe;
  --md-light-blue-100: #b3e5fc;
  --md-light-blue-200: #81d4fa;
  --md-light-blue-300: #4fc3f7;
  --md-light-blue-400: #29b6f6;
  --md-light-blue-500: #03a9f4;
  --md-light-blue-600: #039be5;
  --md-light-blue-700: #0288d1;
  --md-light-blue-800: #0277bd;
  --md-light-blue-900: #01579b;
  --md-light-blue-A100: #80d8ff;
  --md-light-blue-A200: #40c4ff;
  --md-light-blue-A400: #00b0ff;
  --md-light-blue-A700: #0091ea;
  --md-cyan-50: #e0f7fa;
  --md-cyan-100: #b2ebf2;
  --md-cyan-200: #80deea;
  --md-cyan-300: #4dd0e1;
  --md-cyan-400: #26c6da;
  --md-cyan-500: #00bcd4;
  --md-cyan-600: #00acc1;
  --md-cyan-700: #0097a7;
  --md-cyan-800: #00838f;
  --md-cyan-900: #006064;
  --md-cyan-A100: #84ffff;
  --md-cyan-A200: #18ffff;
  --md-cyan-A400: #00e5ff;
  --md-cyan-A700: #00b8d4;
  --md-teal-50: #e0f2f1;
  --md-teal-100: #b2dfdb;
  --md-teal-200: #80cbc4;
  --md-teal-300: #4db6ac;
  --md-teal-400: #26a69a;
  --md-teal-500: #009688;
  --md-teal-600: #00897b;
  --md-teal-700: #00796b;
  --md-teal-800: #00695c;
  --md-teal-900: #004d40;
  --md-teal-A100: #a7ffeb;
  --md-teal-A200: #64ffda;
  --md-teal-A400: #1de9b6;
  --md-teal-A700: #00bfa5;
  --md-green-50: #e8f5e9;
  --md-green-100: #c8e6c9;
  --md-green-200: #a5d6a7;
  --md-green-300: #81c784;
  --md-green-400: #66bb6a;
  --md-green-500: #4caf50;
  --md-green-600: #43a047;
  --md-green-700: #388e3c;
  --md-green-800: #2e7d32;
  --md-green-900: #1b5e20;
  --md-green-A100: #b9f6ca;
  --md-green-A200: #69f0ae;
  --md-green-A400: #00e676;
  --md-green-A700: #00c853;
  --md-light-green-50: #f1f8e9;
  --md-light-green-100: #dcedc8;
  --md-light-green-200: #c5e1a5;
  --md-light-green-300: #aed581;
  --md-light-green-400: #9ccc65;
  --md-light-green-500: #8bc34a;
  --md-light-green-600: #7cb342;
  --md-light-green-700: #689f38;
  --md-light-green-800: #558b2f;
  --md-light-green-900: #33691e;
  --md-light-green-A100: #ccff90;
  --md-light-green-A200: #b2ff59;
  --md-light-green-A400: #76ff03;
  --md-light-green-A700: #64dd17;
  --md-lime-50: #f9fbe7;
  --md-lime-100: #f0f4c3;
  --md-lime-200: #e6ee9c;
  --md-lime-300: #dce775;
  --md-lime-400: #d4e157;
  --md-lime-500: #cddc39;
  --md-lime-600: #c0ca33;
  --md-lime-700: #afb42b;
  --md-lime-800: #9e9d24;
  --md-lime-900: #827717;
  --md-lime-A100: #f4ff81;
  --md-lime-A200: #eeff41;
  --md-lime-A400: #c6ff00;
  --md-lime-A700: #aeea00;
  --md-yellow-50: #fffde7;
  --md-yellow-100: #fff9c4;
  --md-yellow-200: #fff59d;
  --md-yellow-300: #fff176;
  --md-yellow-400: #ffee58;
  --md-yellow-500: #ffeb3b;
  --md-yellow-600: #fdd835;
  --md-yellow-700: #fbc02d;
  --md-yellow-800: #f9a825;
  --md-yellow-900: #f57f17;
  --md-yellow-A100: #ffff8d;
  --md-yellow-A200: #ff0;
  --md-yellow-A400: #ffea00;
  --md-yellow-A700: #ffd600;
  --md-amber-50: #fff8e1;
  --md-amber-100: #ffecb3;
  --md-amber-200: #ffe082;
  --md-amber-300: #ffd54f;
  --md-amber-400: #ffca28;
  --md-amber-500: #ffc107;
  --md-amber-600: #ffb300;
  --md-amber-700: #ffa000;
  --md-amber-800: #ff8f00;
  --md-amber-900: #ff6f00;
  --md-amber-A100: #ffe57f;
  --md-amber-A200: #ffd740;
  --md-amber-A400: #ffc400;
  --md-amber-A700: #ffab00;
  --md-orange-50: #fff3e0;
  --md-orange-100: #ffe0b2;
  --md-orange-200: #ffcc80;
  --md-orange-300: #ffb74d;
  --md-orange-400: #ffa726;
  --md-orange-500: #ff9800;
  --md-orange-600: #fb8c00;
  --md-orange-700: #f57c00;
  --md-orange-800: #ef6c00;
  --md-orange-900: #e65100;
  --md-orange-A100: #ffd180;
  --md-orange-A200: #ffab40;
  --md-orange-A400: #ff9100;
  --md-orange-A700: #ff6d00;
  --md-deep-orange-50: #fbe9e7;
  --md-deep-orange-100: #ffccbc;
  --md-deep-orange-200: #ffab91;
  --md-deep-orange-300: #ff8a65;
  --md-deep-orange-400: #ff7043;
  --md-deep-orange-500: #ff5722;
  --md-deep-orange-600: #f4511e;
  --md-deep-orange-700: #e64a19;
  --md-deep-orange-800: #d84315;
  --md-deep-orange-900: #bf360c;
  --md-deep-orange-A100: #ff9e80;
  --md-deep-orange-A200: #ff6e40;
  --md-deep-orange-A400: #ff3d00;
  --md-deep-orange-A700: #dd2c00;
  --md-brown-50: #efebe9;
  --md-brown-100: #d7ccc8;
  --md-brown-200: #bcaaa4;
  --md-brown-300: #a1887f;
  --md-brown-400: #8d6e63;
  --md-brown-500: #795548;
  --md-brown-600: #6d4c41;
  --md-brown-700: #5d4037;
  --md-brown-800: #4e342e;
  --md-brown-900: #3e2723;
  --md-grey-50: #fafafa;
  --md-grey-100: #f5f5f5;
  --md-grey-200: #eee;
  --md-grey-300: #e0e0e0;
  --md-grey-400: #bdbdbd;
  --md-grey-500: #9e9e9e;
  --md-grey-600: #757575;
  --md-grey-700: #616161;
  --md-grey-800: #424242;
  --md-grey-900: #212121;
  --md-blue-grey-50: #eceff1;
  --md-blue-grey-100: #cfd8dc;
  --md-blue-grey-200: #b0bec5;
  --md-blue-grey-300: #90a4ae;
  --md-blue-grey-400: #78909c;
  --md-blue-grey-500: #607d8b;
  --md-blue-grey-600: #546e7a;
  --md-blue-grey-700: #455a64;
  --md-blue-grey-800: #37474f;
  --md-blue-grey-900: #263238;
}

/*-----------------------------------------------------------------------------
| Copyright (c) 2014-2017, Jupyter Development Team.
|
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| RenderedText
|----------------------------------------------------------------------------*/

:root {
  /* This is the padding value to fill the gaps between lines containing spans with background color. */
  --jp-private-code-span-padding: calc(
    (var(--jp-code-line-height) - 1) * var(--jp-code-font-size) / 2
  );
}

.jp-RenderedText {
  text-align: left;
  padding-left: var(--jp-code-padding);
  line-height: var(--jp-code-line-height);
  font-family: var(--jp-code-font-family);
}

.jp-RenderedText pre,
.jp-RenderedJavaScript pre,
.jp-RenderedHTMLCommon pre {
  color: var(--jp-content-font-color1);
  font-size: var(--jp-code-font-size);
  border: none;
  margin: 0;
  padding: 0;
}

.jp-RenderedText pre a:link {
  text-decoration: none;
  color: var(--jp-content-link-color);
}

.jp-RenderedText pre a:hover {
  text-decoration: underline;
  color: var(--jp-content-link-color);
}

.jp-RenderedText pre a:visited {
  text-decoration: none;
  color: var(--jp-content-link-color);
}

/* console foregrounds and backgrounds */
.jp-RenderedText pre .ansi-black-fg {
  color: #3e424d;
}

.jp-RenderedText pre .ansi-red-fg {
  color: #e75c58;
}

.jp-RenderedText pre .ansi-green-fg {
  color: #00a250;
}

.jp-RenderedText pre .ansi-yellow-fg {
  color: #ddb62b;
}

.jp-RenderedText pre .ansi-blue-fg {
  color: #208ffb;
}

.jp-RenderedText pre .ansi-magenta-fg {
  color: #d160c4;
}

.jp-RenderedText pre .ansi-cyan-fg {
  color: #60c6c8;
}

.jp-RenderedText pre .ansi-white-fg {
  color: #c5c1b4;
}

.jp-RenderedText pre .ansi-black-bg {
  background-color: #3e424d;
  padding: var(--jp-private-code-span-padding) 0;
}

.jp-RenderedText pre .ansi-red-bg {
  background-color: #e75c58;
  padding: var(--jp-private-code-span-padding) 0;
}

.jp-RenderedText pre .ansi-green-bg {
  background-color: #00a250;
  padding: var(--jp-private-code-span-padding) 0;
}

.jp-RenderedText pre .ansi-yellow-bg {
  background-color: #ddb62b;
  padding: var(--jp-private-code-span-padding) 0;
}

.jp-RenderedText pre .ansi-blue-bg {
  background-color: #208ffb;
  padding: var(--jp-private-code-span-padding) 0;
}

.jp-RenderedText pre .ansi-magenta-bg {
  background-color: #d160c4;
  padding: var(--jp-private-code-span-padding) 0;
}

.jp-RenderedText pre .ansi-cyan-bg {
  background-color: #60c6c8;
  padding: var(--jp-private-code-span-padding) 0;
}

.jp-RenderedText pre .ansi-white-bg {
  background-color: #c5c1b4;
  padding: var(--jp-private-code-span-padding) 0;
}

.jp-RenderedText pre .ansi-black-intense-fg {
  color: #282c36;
}

.jp-RenderedText pre .ansi-red-intense-fg {
  color: #b22b31;
}

.jp-RenderedText pre .ansi-green-intense-fg {
  color: #007427;
}

.jp-RenderedText pre .ansi-yellow-intense-fg {
  color: #b27d12;
}

.jp-RenderedText pre .ansi-blue-intense-fg {
  color: #0065ca;
}

.jp-RenderedText pre .ansi-magenta-intense-fg {
  color: #a03196;
}

.jp-RenderedText pre .ansi-cyan-intense-fg {
  color: #258f8f;
}

.jp-RenderedText pre .ansi-white-intense-fg {
  color: #a1a6b2;
}

.jp-RenderedText pre .ansi-black-intense-bg {
  background-color: #282c36;
  padding: var(--jp-private-code-span-padding) 0;
}

.jp-RenderedText pre .ansi-red-intense-bg {
  background-color: #b22b31;
  padding: var(--jp-private-code-span-padding) 0;
}

.jp-RenderedText pre .ansi-green-intense-bg {
  background-color: #007427;
  padding: var(--jp-private-code-span-padding) 0;
}

.jp-RenderedText pre .ansi-yellow-intense-bg {
  background-color: #b27d12;
  padding: var(--jp-private-code-span-padding) 0;
}

.jp-RenderedText pre .ansi-blue-intense-bg {
  background-color: #0065ca;
  padding: var(--jp-private-code-span-padding) 0;
}

.jp-RenderedText pre .ansi-magenta-intense-bg {
  background-color: #a03196;
  padding: var(--jp-private-code-span-padding) 0;
}

.jp-RenderedText pre .ansi-cyan-intense-bg {
  background-color: #258f8f;
  padding: var(--jp-private-code-span-padding) 0;
}

.jp-RenderedText pre .ansi-white-intense-bg {
  background-color: #a1a6b2;
  padding: var(--jp-private-code-span-padding) 0;
}

.jp-RenderedText pre .ansi-default-inverse-fg {
  color: var(--jp-ui-inverse-font-color0);
}

.jp-RenderedText pre .ansi-default-inverse-bg {
  background-color: var(--jp-inverse-layout-color0);
  padding: var(--jp-private-code-span-padding) 0;
}

.jp-RenderedText pre .ansi-bold {
  font-weight: bold;
}

.jp-RenderedText pre .ansi-underline {
  text-decoration: underline;
}

.jp-RenderedText[data-mime-type='application/vnd.jupyter.stderr'] {
  background: var(--jp-rendermime-error-background);
  padding-top: var(--jp-code-padding);
}

/*-----------------------------------------------------------------------------
| RenderedLatex
|----------------------------------------------------------------------------*/

.jp-RenderedLatex {
  color: var(--jp-content-font-color1);
  font-size: var(--jp-content-font-size1);
  line-height: var(--jp-content-line-height);
}

/* Left-justify outputs.*/
.jp-OutputArea-output.jp-RenderedLatex {
  padding: var(--jp-code-padding);
  text-align: left;
}

/*-----------------------------------------------------------------------------
| RenderedHTML
|----------------------------------------------------------------------------*/

.jp-RenderedHTMLCommon {
  color: var(--jp-content-font-color1);
  font-family: var(--jp-content-font-family);
  font-size: var(--jp-content-font-size1);
  line-height: var(--jp-content-line-height);

  /* Give a bit more R padding on Markdown text to keep line lengths reasonable */
  padding-right: 20px;
}

.jp-RenderedHTMLCommon em {
  font-style: italic;
}

.jp-RenderedHTMLCommon strong {
  font-weight: bold;
}

.jp-RenderedHTMLCommon u {
  text-decoration: underline;
}

.jp-RenderedHTMLCommon a:link {
  text-decoration: none;
  color: var(--jp-content-link-color);
}

.jp-RenderedHTMLCommon a:hover {
  text-decoration: underline;
  color: var(--jp-content-link-color);
}

.jp-RenderedHTMLCommon a:visited {
  text-decoration: none;
  color: var(--jp-content-link-color);
}

/* Headings */

.jp-RenderedHTMLCommon h1,
.jp-RenderedHTMLCommon h2,
.jp-RenderedHTMLCommon h3,
.jp-RenderedHTMLCommon h4,
.jp-RenderedHTMLCommon h5,
.jp-RenderedHTMLCommon h6 {
  line-height: var(--jp-content-heading-line-height);
  font-weight: var(--jp-content-heading-font-weight);
  font-style: normal;
  margin: var(--jp-content-heading-margin-top) 0
    var(--jp-content-heading-margin-bottom) 0;
}

.jp-RenderedHTMLCommon h1:first-child,
.jp-RenderedHTMLCommon h2:first-child,
.jp-RenderedHTMLCommon h3:first-child,
.jp-RenderedHTMLCommon h4:first-child,
.jp-RenderedHTMLCommon h5:first-child,
.jp-RenderedHTMLCommon h6:first-child {
  margin-top: calc(0.5 * var(--jp-content-heading-margin-top));
}

.jp-RenderedHTMLCommon h1:last-child,
.jp-RenderedHTMLCommon h2:last-child,
.jp-RenderedHTMLCommon h3:last-child,
.jp-RenderedHTMLCommon h4:last-child,
.jp-RenderedHTMLCommon h5:last-child,
.jp-RenderedHTMLCommon h6:last-child {
  margin-bottom: calc(0.5 * var(--jp-content-heading-margin-bottom));
}

.jp-RenderedHTMLCommon h1 {
  font-size: var(--jp-content-font-size5);
}

.jp-RenderedHTMLCommon h2 {
  font-size: var(--jp-content-font-size4);
}

.jp-RenderedHTMLCommon h3 {
  font-size: var(--jp-content-font-size3);
}

.jp-RenderedHTMLCommon h4 {
  font-size: var(--jp-content-font-size2);
}

.jp-RenderedHTMLCommon h5 {
  font-size: var(--jp-content-font-size1);
}

.jp-RenderedHTMLCommon h6 {
  font-size: var(--jp-content-font-size0);
}

/* Lists */

/* stylelint-disable selector-max-type, selector-max-compound-selectors */

.jp-RenderedHTMLCommon ul:not(.list-inline),
.jp-RenderedHTMLCommon ol:not(.list-inline) {
  padding-left: 2em;
}

.jp-RenderedHTMLCommon ul {
  list-style: disc;
}

.jp-RenderedHTMLCommon ul ul {
  list-style: square;
}

.jp-RenderedHTMLCommon ul ul ul {
  list-style: circle;
}

.jp-RenderedHTMLCommon ol {
  list-style: decimal;
}

.jp-RenderedHTMLCommon ol ol {
  list-style: upper-alpha;
}

.jp-RenderedHTMLCommon ol ol ol {
  list-style: lower-alpha;
}

.jp-RenderedHTMLCommon ol ol ol ol {
  list-style: lower-roman;
}

.jp-RenderedHTMLCommon ol ol ol ol ol {
  list-style: decimal;
}

.jp-RenderedHTMLCommon ol,
.jp-RenderedHTMLCommon ul {
  margin-bottom: 1em;
}

.jp-RenderedHTMLCommon ul ul,
.jp-RenderedHTMLCommon ul ol,
.jp-RenderedHTMLCommon ol ul,
.jp-RenderedHTMLCommon ol ol {
  margin-bottom: 0;
}

/* stylelint-enable selector-max-type, selector-max-compound-selectors */

.jp-RenderedHTMLCommon hr {
  color: var(--jp-border-color2);
  background-color: var(--jp-border-color1);
  margin-top: 1em;
  margin-bottom: 1em;
}

.jp-RenderedHTMLCommon > pre {
  margin: 1.5em 2em;
}

.jp-RenderedHTMLCommon pre,
.jp-RenderedHTMLCommon code {
  border: 0;
  background-color: var(--jp-layout-color0);
  color: var(--jp-content-font-color1);
  font-family: var(--jp-code-font-family);
  font-size: inherit;
  line-height: var(--jp-code-line-height);
  padding: 0;
  white-space: pre-wrap;
}

.jp-RenderedHTMLCommon :not(pre) > code {
  background-color: var(--jp-layout-color2);
  padding: 1px 5px;
}

/* Tables */

.jp-RenderedHTMLCommon table {
  border-collapse: collapse;
  border-spacing: 0;
  border: none;
  color: var(--jp-ui-font-color1);
  font-size: var(--jp-ui-font-size1);
  table-layout: fixed;
  margin-left: auto;
  margin-bottom: 1em;
  margin-right: auto;
}

.jp-RenderedHTMLCommon thead {
  border-bottom: var(--jp-border-width) solid var(--jp-border-color1);
  vertical-align: bottom;
}

.jp-RenderedHTMLCommon td,
.jp-RenderedHTMLCommon th,
.jp-RenderedHTMLCommon tr {
  vertical-align: middle;
  padding: 0.5em;
  line-height: normal;
  white-space: normal;
  max-width: none;
  border: none;
}

.jp-RenderedMarkdown.jp-RenderedHTMLCommon td,
.jp-RenderedMarkdown.jp-RenderedHTMLCommon th {
  max-width: none;
}

:not(.jp-RenderedMarkdown).jp-RenderedHTMLCommon td,
:not(.jp-RenderedMarkdown).jp-RenderedHTMLCommon th,
:not(.jp-RenderedMarkdown).jp-RenderedHTMLCommon tr {
  text-align: right;
}

.jp-RenderedHTMLCommon th {
  font-weight: bold;
}

.jp-RenderedHTMLCommon tbody tr:nth-child(odd) {
  background: var(--jp-layout-color0);
}

.jp-RenderedHTMLCommon tbody tr:nth-child(even) {
  background: var(--jp-rendermime-table-row-background);
}

.jp-RenderedHTMLCommon tbody tr:hover {
  background: var(--jp-rendermime-table-row-hover-background);
}

.jp-RenderedHTMLCommon p {
  text-align: left;
  margin: 0;
  margin-bottom: 1em;
}

.jp-RenderedHTMLCommon img {
  -moz-force-broken-image-icon: 1;
}

/* Restrict to direct children as other images could be nested in other content. */
.jp-RenderedHTMLCommon > img {
  display: block;
  margin-left: 0;
  margin-right: 0;
  margin-bottom: 1em;
}

/* Change color behind transparent images if they need it... */
[data-jp-theme-light='false'] .jp-RenderedImage img.jp-needs-light-background {
  background-color: var(--jp-inverse-layout-color1);
}

[data-jp-theme-light='true'] .jp-RenderedImage img.jp-needs-dark-background {
  background-color: var(--jp-inverse-layout-color1);
}

.jp-RenderedHTMLCommon img,
.jp-RenderedImage img,
.jp-RenderedHTMLCommon svg,
.jp-RenderedSVG svg {
  max-width: 100%;
  height: auto;
}

.jp-RenderedHTMLCommon img.jp-mod-unconfined,
.jp-RenderedImage img.jp-mod-unconfined,
.jp-RenderedHTMLCommon svg.jp-mod-unconfined,
.jp-RenderedSVG svg.jp-mod-unconfined {
  max-width: none;
}

.jp-RenderedHTMLCommon .alert {
  padding: var(--jp-notebook-padding);
  border: var(--jp-border-width) solid transparent;
  border-radius: var(--jp-border-radius);
  margin-bottom: 1em;
}

.jp-RenderedHTMLCommon .alert-info {
  color: var(--jp-info-color0);
  background-color: var(--jp-info-color3);
  border-color: var(--jp-info-color2);
}

.jp-RenderedHTMLCommon .alert-info hr {
  border-color: var(--jp-info-color3);
}

.jp-RenderedHTMLCommon .alert-info > p:last-child,
.jp-RenderedHTMLCommon .alert-info > ul:last-child {
  margin-bottom: 0;
}

.jp-RenderedHTMLCommon .alert-warning {
  color: var(--jp-warn-color0);
  background-color: var(--jp-warn-color3);
  border-color: var(--jp-warn-color2);
}

.jp-RenderedHTMLCommon .alert-warning hr {
  border-color: var(--jp-warn-color3);
}

.jp-RenderedHTMLCommon .alert-warning > p:last-child,
.jp-RenderedHTMLCommon .alert-warning > ul:last-child {
  margin-bottom: 0;
}

.jp-RenderedHTMLCommon .alert-success {
  color: var(--jp-success-color0);
  background-color: var(--jp-success-color3);
  border-color: var(--jp-success-color2);
}

.jp-RenderedHTMLCommon .alert-success hr {
  border-color: var(--jp-success-color3);
}

.jp-RenderedHTMLCommon .alert-success > p:last-child,
.jp-RenderedHTMLCommon .alert-success > ul:last-child {
  margin-bottom: 0;
}

.jp-RenderedHTMLCommon .alert-danger {
  color: var(--jp-error-color0);
  background-color: var(--jp-error-color3);
  border-color: var(--jp-error-color2);
}

.jp-RenderedHTMLCommon .alert-danger hr {
  border-color: var(--jp-error-color3);
}

.jp-RenderedHTMLCommon .alert-danger > p:last-child,
.jp-RenderedHTMLCommon .alert-danger > ul:last-child {
  margin-bottom: 0;
}

.jp-RenderedHTMLCommon blockquote {
  margin: 1em 2em;
  padding: 0 1em;
  border-left: 5px solid var(--jp-border-color2);
}

a.jp-InternalAnchorLink {
  visibility: hidden;
  margin-left: 8px;
  color: var(--md-blue-800);
}

h1:hover .jp-InternalAnchorLink,
h2:hover .jp-InternalAnchorLink,
h3:hover .jp-InternalAnchorLink,
h4:hover .jp-InternalAnchorLink,
h5:hover .jp-InternalAnchorLink,
h6:hover .jp-InternalAnchorLink {
  visibility: visible;
}

.jp-RenderedHTMLCommon kbd {
  background-color: var(--jp-rendermime-table-row-background);
  border: 1px solid var(--jp-border-color0);
  border-bottom-color: var(--jp-border-color2);
  border-radius: 3px;
  box-shadow: inset 0 -1px 0 rgba(0, 0, 0, 0.25);
  display: inline-block;
  font-size: var(--jp-ui-font-size0);
  line-height: 1em;
  padding: 0.2em 0.5em;
}

/* Most direct children of .jp-RenderedHTMLCommon have a margin-bottom of 1.0.
 * At the bottom of cells this is a bit too much as there is also spacing
 * between cells. Going all the way to 0 gets too tight between markdown and
 * code cells.
 */
.jp-RenderedHTMLCommon > *:last-child {
  margin-bottom: 0.5em;
}

/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Copyright (c) 2014-2017, PhosphorJS Contributors
|
| Distributed under the terms of the BSD 3-Clause License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/

.lm-cursor-backdrop {
  position: fixed;
  width: 200px;
  height: 200px;
  margin-top: -100px;
  margin-left: -100px;
  will-change: transform;
  z-index: 100;
}

.lm-mod-drag-image {
  will-change: transform;
}

/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

.jp-lineFormSearch {
  padding: 4px 12px;
  background-color: var(--jp-layout-color2);
  box-shadow: var(--jp-toolbar-box-shadow);
  z-index: 2;
  font-size: var(--jp-ui-font-size1);
}

.jp-lineFormCaption {
  font-size: var(--jp-ui-font-size0);
  line-height: var(--jp-ui-font-size1);
  margin-top: 4px;
  color: var(--jp-ui-font-color0);
}

.jp-baseLineForm {
  border: none;
  border-radius: 0;
  position: absolute;
  background-size: 16px;
  background-repeat: no-repeat;
  background-position: center;
  outline: none;
}

.jp-lineFormButtonContainer {
  top: 4px;
  right: 8px;
  height: 24px;
  padding: 0 12px;
  width: 12px;
}

.jp-lineFormButtonIcon {
  top: 0;
  right: 0;
  background-color: var(--jp-brand-color1);
  height: 100%;
  width: 100%;
  box-sizing: border-box;
  padding: 4px 6px;
}

.jp-lineFormButton {
  top: 0;
  right: 0;
  background-color: transparent;
  height: 100%;
  width: 100%;
  box-sizing: border-box;
}

.jp-lineFormWrapper {
  overflow: hidden;
  padding: 0 8px;
  border: 1px solid var(--jp-border-color0);
  background-color: var(--jp-input-active-background);
  height: 22px;
}

.jp-lineFormWrapperFocusWithin {
  border: var(--jp-border-width) solid var(--md-blue-500);
  box-shadow: inset 0 0 4px var(--md-blue-300);
}

.jp-lineFormInput {
  background: transparent;
  width: 200px;
  height: 100%;
  border: none;
  outline: none;
  color: var(--jp-ui-font-color0);
  line-height: 28px;
}

/*-----------------------------------------------------------------------------
| Copyright (c) 2014-2016, Jupyter Development Team.
|
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-JSONEditor {
  display: flex;
  flex-direction: column;
  width: 100%;
}

.jp-JSONEditor-host {
  flex: 1 1 auto;
  border: var(--jp-border-width) solid var(--jp-input-border-color);
  border-radius: 0;
  background: var(--jp-layout-color0);
  min-height: 50px;
  padding: 1px;
}

.jp-JSONEditor.jp-mod-error .jp-JSONEditor-host {
  border-color: red;
  outline-color: red;
}

.jp-JSONEditor-header {
  display: flex;
  flex: 1 0 auto;
  padding: 0 0 0 12px;
}

.jp-JSONEditor-header label {
  flex: 0 0 auto;
}

.jp-JSONEditor-commitButton {
  height: 16px;
  width: 16px;
  background-size: 18px;
  background-repeat: no-repeat;
  background-position: center;
}

.jp-JSONEditor-host.jp-mod-focused {
  background-color: var(--jp-input-active-background);
  border: 1px solid var(--jp-input-active-border-color);
  box-shadow: var(--jp-input-box-shadow);
}

.jp-Editor.jp-mod-dropTarget {
  border: var(--jp-border-width) solid var(--jp-input-active-border-color);
  box-shadow: var(--jp-input-box-shadow);
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/
.jp-DocumentSearch-input {
  border: none;
  outline: none;
  color: var(--jp-ui-font-color0);
  font-size: var(--jp-ui-font-size1);
  background-color: var(--jp-layout-color0);
  font-family: var(--jp-ui-font-family);
  padding: 2px 1px;
  resize: none;
}

.jp-DocumentSearch-overlay {
  position: absolute;
  background-color: var(--jp-toolbar-background);
  border-bottom: var(--jp-border-width) solid var(--jp-toolbar-border-color);
  border-left: var(--jp-border-width) solid var(--jp-toolbar-border-color);
  top: 0;
  right: 0;
  z-index: 7;
  min-width: 405px;
  padding: 2px;
  font-size: var(--jp-ui-font-size1);

  --jp-private-document-search-button-height: 20px;
}

.jp-DocumentSearch-overlay button {
  background-color: var(--jp-toolbar-background);
  outline: 0;
}

.jp-DocumentSearch-overlay button:hover {
  background-color: var(--jp-layout-color2);
}

.jp-DocumentSearch-overlay button:active {
  background-color: var(--jp-layout-color3);
}

.jp-DocumentSearch-overlay-row {
  display: flex;
  align-items: center;
  margin-bottom: 2px;
}

.jp-DocumentSearch-button-content {
  display: inline-block;
  cursor: pointer;
  box-sizing: border-box;
  width: 100%;
  height: 100%;
}

.jp-DocumentSearch-button-content svg {
  width: 100%;
  height: 100%;
}

.jp-DocumentSearch-input-wrapper {
  border: var(--jp-border-width) solid var(--jp-border-color0);
  display: flex;
  background-color: var(--jp-layout-color0);
  margin: 2px;
}

.jp-DocumentSearch-input-wrapper:focus-within {
  border-color: var(--jp-cell-editor-active-border-color);
}

.jp-DocumentSearch-toggle-wrapper,
.jp-DocumentSearch-button-wrapper {
  all: initial;
  overflow: hidden;
  display: inline-block;
  border: none;
  box-sizing: border-box;
}

.jp-DocumentSearch-toggle-wrapper {
  width: 14px;
  height: 14px;
}

.jp-DocumentSearch-button-wrapper {
  width: var(--jp-private-document-search-button-height);
  height: var(--jp-private-document-search-button-height);
}

.jp-DocumentSearch-toggle-wrapper:focus,
.jp-DocumentSearch-button-wrapper:focus {
  outline: var(--jp-border-width) solid
    var(--jp-cell-editor-active-border-color);
  outline-offset: -1px;
}

.jp-DocumentSearch-toggle-wrapper,
.jp-DocumentSearch-button-wrapper,
.jp-DocumentSearch-button-content:focus {
  outline: none;
}

.jp-DocumentSearch-toggle-placeholder {
  width: 5px;
}

.jp-DocumentSearch-input-button::before {
  display: block;
  padding-top: 100%;
}

.jp-DocumentSearch-input-button-off {
  opacity: var(--jp-search-toggle-off-opacity);
}

.jp-DocumentSearch-input-button-off:hover {
  opacity: var(--jp-search-toggle-hover-opacity);
}

.jp-DocumentSearch-input-button-on {
  opacity: var(--jp-search-toggle-on-opacity);
}

.jp-DocumentSearch-index-counter {
  padding-left: 10px;
  padding-right: 10px;
  user-select: none;
  min-width: 35px;
  display: inline-block;
}

.jp-DocumentSearch-up-down-wrapper {
  display: inline-block;
  padding-right: 2px;
  margin-left: auto;
  white-space: nowrap;
}

.jp-DocumentSearch-spacer {
  margin-left: auto;
}

.jp-DocumentSearch-up-down-wrapper button {
  outline: 0;
  border: none;
  width: var(--jp-private-document-search-button-height);
  height: var(--jp-private-document-search-button-height);
  vertical-align: middle;
  margin: 1px 5px 2px;
}

.jp-DocumentSearch-up-down-button:hover {
  background-color: var(--jp-layout-color2);
}

.jp-DocumentSearch-up-down-button:active {
  background-color: var(--jp-layout-color3);
}

.jp-DocumentSearch-filter-button {
  border-radius: var(--jp-border-radius);
}

.jp-DocumentSearch-filter-button:hover {
  background-color: var(--jp-layout-color2);
}

.jp-DocumentSearch-filter-button-enabled {
  background-color: var(--jp-layout-color2);
}

.jp-DocumentSearch-filter-button-enabled:hover {
  background-color: var(--jp-layout-color3);
}

.jp-DocumentSearch-search-options {
  padding: 0 8px;
  margin-left: 3px;
  width: 100%;
  display: grid;
  justify-content: start;
  grid-template-columns: 1fr 1fr;
  align-items: center;
  justify-items: stretch;
}

.jp-DocumentSearch-search-filter-disabled {
  color: var(--jp-ui-font-color2);
}

.jp-DocumentSearch-search-filter {
  display: flex;
  align-items: center;
  user-select: none;
}

.jp-DocumentSearch-regex-error {
  color: var(--jp-error-color0);
}

.jp-DocumentSearch-replace-button-wrapper {
  overflow: hidden;
  display: inline-block;
  box-sizing: border-box;
  border: var(--jp-border-width) solid var(--jp-border-color0);
  margin: auto 2px;
  padding: 1px 4px;
  height: calc(var(--jp-private-document-search-button-height) + 2px);
}

.jp-DocumentSearch-replace-button-wrapper:focus {
  border: var(--jp-border-width) solid var(--jp-cell-editor-active-border-color);
}

.jp-DocumentSearch-replace-button {
  display: inline-block;
  text-align: center;
  cursor: pointer;
  box-sizing: border-box;
  color: var(--jp-ui-font-color1);

  /* height - 2 * (padding of wrapper) */
  line-height: calc(var(--jp-private-document-search-button-height) - 2px);
  width: 100%;
  height: 100%;
}

.jp-DocumentSearch-replace-button:focus {
  outline: none;
}

.jp-DocumentSearch-replace-wrapper-class {
  margin-left: 14px;
  display: flex;
}

.jp-DocumentSearch-replace-toggle {
  border: none;
  background-color: var(--jp-toolbar-background);
  border-radius: var(--jp-border-radius);
}

.jp-DocumentSearch-replace-toggle:hover {
  background-color: var(--jp-layout-color2);
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.cm-editor {
  line-height: var(--jp-code-line-height);
  font-size: var(--jp-code-font-size);
  font-family: var(--jp-code-font-family);
  border: 0;
  border-radius: 0;
  height: auto;

  /* Changed to auto to autogrow */
}

.cm-editor pre {
  padding: 0 var(--jp-code-padding);
}

.jp-CodeMirrorEditor[data-type='inline'] .cm-dialog {
  background-color: var(--jp-layout-color0);
  color: var(--jp-content-font-color1);
}

.jp-CodeMirrorEditor {
  cursor: text;
}

/* When zoomed out 67% and 33% on a screen of 1440 width x 900 height */
@media screen and (min-width: 2138px) and (max-width: 4319px) {
  .jp-CodeMirrorEditor[data-type='inline'] .cm-cursor {
    border-left: var(--jp-code-cursor-width1) solid
      var(--jp-editor-cursor-color);
  }
}

/* When zoomed out less than 33% */
@media screen and (min-width: 4320px) {
  .jp-CodeMirrorEditor[data-type='inline'] .cm-cursor {
    border-left: var(--jp-code-cursor-width2) solid
      var(--jp-editor-cursor-color);
  }
}

.cm-editor.jp-mod-readOnly .cm-cursor {
  display: none;
}

.jp-CollaboratorCursor {
  border-left: 5px solid transparent;
  border-right: 5px solid transparent;
  border-top: none;
  border-bottom: 3px solid;
  background-clip: content-box;
  margin-left: -5px;
  margin-right: -5px;
}

.cm-searching,
.cm-searching span {
  /* `.cm-searching span`: we need to override syntax highlighting */
  background-color: var(--jp-search-unselected-match-background-color);
  color: var(--jp-search-unselected-match-color);
}

.cm-searching::selection,
.cm-searching span::selection {
  background-color: var(--jp-search-unselected-match-background-color);
  color: var(--jp-search-unselected-match-color);
}

.jp-current-match > .cm-searching,
.jp-current-match > .cm-searching span,
.cm-searching > .jp-current-match,
.cm-searching > .jp-current-match span {
  background-color: var(--jp-search-selected-match-background-color);
  color: var(--jp-search-selected-match-color);
}

.jp-current-match > .cm-searching::selection,
.cm-searching > .jp-current-match::selection,
.jp-current-match > .cm-searching span::selection {
  background-color: var(--jp-search-selected-match-background-color);
  color: var(--jp-search-selected-match-color);
}

.cm-trailingspace {
  background-image: url(data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAgAAAAFCAYAAAB4ka1VAAAAsElEQVQIHQGlAFr/AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA7+r3zKmT0/+pk9P/7+r3zAAAAAAAAAAABAAAAAAAAAAA6OPzM+/q9wAAAAAA6OPzMwAAAAAAAAAAAgAAAAAAAAAAGR8NiRQaCgAZIA0AGR8NiQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAQyoYJ/SY80UAAAAASUVORK5CYII=);
  background-position: center left;
  background-repeat: repeat-x;
}

.jp-CollaboratorCursor-hover {
  position: absolute;
  z-index: 1;
  transform: translateX(-50%);
  color: white;
  border-radius: 3px;
  padding-left: 4px;
  padding-right: 4px;
  padding-top: 1px;
  padding-bottom: 1px;
  text-align: center;
  font-size: var(--jp-ui-font-size1);
  white-space: nowrap;
}

.jp-CodeMirror-ruler {
  border-left: 1px dashed var(--jp-border-color2);
}

/* Styles for shared cursors (remote cursor locations and selected ranges) */
.jp-CodeMirrorEditor .cm-ySelectionCaret {
  position: relative;
  border-left: 1px solid black;
  margin-left: -1px;
  margin-right: -1px;
  box-sizing: border-box;
}

.jp-CodeMirrorEditor .cm-ySelectionCaret > .cm-ySelectionInfo {
  white-space: nowrap;
  position: absolute;
  top: -1.15em;
  padding-bottom: 0.05em;
  left: -1px;
  font-size: 0.95em;
  font-family: var(--jp-ui-font-family);
  font-weight: bold;
  line-height: normal;
  user-select: none;
  color: white;
  padding-left: 2px;
  padding-right: 2px;
  z-index: 101;
  transition: opacity 0.3s ease-in-out;
}

.jp-CodeMirrorEditor .cm-ySelectionInfo {
  transition-delay: 0.7s;
  opacity: 0;
}

.jp-CodeMirrorEditor .cm-ySelectionCaret:hover > .cm-ySelectionInfo {
  opacity: 1;
  transition-delay: 0s;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-MimeDocument {
  outline: none;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Variables
|----------------------------------------------------------------------------*/

:root {
  --jp-private-filebrowser-button-height: 28px;
  --jp-private-filebrowser-button-width: 48px;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-FileBrowser .jp-SidePanel-content {
  display: flex;
  flex-direction: column;
}

.jp-FileBrowser-toolbar.jp-Toolbar {
  flex-wrap: wrap;
  row-gap: 12px;
  border-bottom: none;
  height: auto;
  margin: 8px 12px 0;
  box-shadow: none;
  padding: 0;
  justify-content: flex-start;
}

.jp-FileBrowser-Panel {
  flex: 1 1 auto;
  display: flex;
  flex-direction: column;
}

.jp-BreadCrumbs {
  flex: 0 0 auto;
  margin: 8px 12px;
}

.jp-BreadCrumbs-item {
  margin: 0 2px;
  padding: 0 2px;
  border-radius: var(--jp-border-radius);
  cursor: pointer;
}

.jp-BreadCrumbs-item:hover {
  background-color: var(--jp-layout-color2);
}

.jp-BreadCrumbs-item:first-child {
  margin-left: 0;
}

.jp-BreadCrumbs-item.jp-mod-dropTarget {
  background-color: var(--jp-brand-color2);
  opacity: 0.7;
}

/*-----------------------------------------------------------------------------
| Buttons
|----------------------------------------------------------------------------*/

.jp-FileBrowser-toolbar > .jp-Toolbar-item {
  flex: 0 0 auto;
  padding-left: 0;
  padding-right: 2px;
  align-items: center;
  height: unset;
}

.jp-FileBrowser-toolbar > .jp-Toolbar-item .jp-ToolbarButtonComponent {
  width: 40px;
}

/*-----------------------------------------------------------------------------
| Other styles
|----------------------------------------------------------------------------*/

.jp-FileDialog.jp-mod-conflict input {
  color: var(--jp-error-color1);
}

.jp-FileDialog .jp-new-name-title {
  margin-top: 12px;
}

.jp-LastModified-hidden {
  display: none;
}

.jp-FileSize-hidden {
  display: none;
}

.jp-FileBrowser .lm-AccordionPanel > h3:first-child {
  display: none;
}

/*-----------------------------------------------------------------------------
| DirListing
|----------------------------------------------------------------------------*/

.jp-DirListing {
  flex: 1 1 auto;
  display: flex;
  flex-direction: column;
  outline: 0;
}

.jp-DirListing-header {
  flex: 0 0 auto;
  display: flex;
  flex-direction: row;
  align-items: center;
  overflow: hidden;
  border-top: var(--jp-border-width) solid var(--jp-border-color2);
  border-bottom: var(--jp-border-width) solid var(--jp-border-color1);
  box-shadow: var(--jp-toolbar-box-shadow);
  z-index: 2;
}

.jp-DirListing-headerItem {
  padding: 4px 12px 2px;
  font-weight: 500;
}

.jp-DirListing-headerItem:hover {
  background: var(--jp-layout-color2);
}

.jp-DirListing-headerItem.jp-id-name {
  flex: 1 0 84px;
}

.jp-DirListing-headerItem.jp-id-modified {
  flex: 0 0 112px;
  border-left: var(--jp-border-width) solid var(--jp-border-color2);
  text-align: right;
}

.jp-DirListing-headerItem.jp-id-filesize {
  flex: 0 0 75px;
  border-left: var(--jp-border-width) solid var(--jp-border-color2);
  text-align: right;
}

.jp-id-narrow {
  display: none;
  flex: 0 0 5px;
  padding: 4px;
  border-left: var(--jp-border-width) solid var(--jp-border-color2);
  text-align: right;
  color: var(--jp-border-color2);
}

.jp-DirListing-narrow .jp-id-narrow {
  display: block;
}

.jp-DirListing-narrow .jp-id-modified,
.jp-DirListing-narrow .jp-DirListing-itemModified {
  display: none;
}

.jp-DirListing-headerItem.jp-mod-selected {
  font-weight: 600;
}

/* increase specificity to override bundled default */
.jp-DirListing-content {
  flex: 1 1 auto;
  margin: 0;
  padding: 0;
  list-style-type: none;
  overflow: auto;
  background-color: var(--jp-layout-color1);
}

.jp-DirListing-content mark {
  color: var(--jp-ui-font-color0);
  background-color: transparent;
  font-weight: bold;
}

.jp-DirListing-content .jp-DirListing-item.jp-mod-selected mark {
  color: var(--jp-ui-inverse-font-color0);
}

/* Style the directory listing content when a user drops a file to upload */
.jp-DirListing.jp-mod-native-drop .jp-DirListing-content {
  outline: 5px dashed rgba(128, 128, 128, 0.5);
  outline-offset: -10px;
  cursor: copy;
}

.jp-DirListing-item {
  display: flex;
  flex-direction: row;
  align-items: center;
  padding: 4px 12px;
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
}

.jp-DirListing-checkboxWrapper {
  /* Increases hit area of checkbox. */
  padding: 4px;
}

.jp-DirListing-header
  .jp-DirListing-checkboxWrapper
  + .jp-DirListing-headerItem {
  padding-left: 4px;
}

.jp-DirListing-content .jp-DirListing-checkboxWrapper {
  position: relative;
  left: -4px;
  margin: -4px 0 -4px -8px;
}

.jp-DirListing-checkboxWrapper.jp-mod-visible {
  visibility: visible;
}

/* For devices that support hovering, hide checkboxes until hovered, selected...
*/
@media (hover: hover) {
  .jp-DirListing-checkboxWrapper {
    visibility: hidden;
  }

  .jp-DirListing-item:hover .jp-DirListing-checkboxWrapper,
  .jp-DirListing-item.jp-mod-selected .jp-DirListing-checkboxWrapper {
    visibility: visible;
  }
}

.jp-DirListing-item[data-is-dot] {
  opacity: 75%;
}

.jp-DirListing-item.jp-mod-selected {
  color: var(--jp-ui-inverse-font-color1);
  background: var(--jp-brand-color1);
}

.jp-DirListing-item.jp-mod-dropTarget {
  background: var(--jp-brand-color3);
}

.jp-DirListing-item:hover:not(.jp-mod-selected) {
  background: var(--jp-layout-color2);
}

.jp-DirListing-itemIcon {
  flex: 0 0 20px;
  margin-right: 4px;
}

.jp-DirListing-itemText {
  flex: 1 0 64px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  user-select: none;
}

.jp-DirListing-itemText:focus {
  outline-width: 2px;
  outline-color: var(--jp-inverse-layout-color1);
  outline-style: solid;
  outline-offset: 1px;
}

.jp-DirListing-item.jp-mod-selected .jp-DirListing-itemText:focus {
  outline-color: var(--jp-layout-color1);
}

.jp-DirListing-itemModified {
  flex: 0 0 125px;
  text-align: right;
}

.jp-DirListing-itemFileSize {
  flex: 0 0 90px;
  text-align: right;
}

.jp-DirListing-editor {
  flex: 1 0 64px;
  outline: none;
  border: none;
  color: var(--jp-ui-font-color1);
  background-color: var(--jp-layout-color1);
}

.jp-DirListing-item.jp-mod-running .jp-DirListing-itemIcon::before {
  color: var(--jp-success-color1);
  content: '\25CF';
  font-size: 8px;
  position: absolute;
  left: -8px;
}

.jp-DirListing-item.jp-mod-running.jp-mod-selected
  .jp-DirListing-itemIcon::before {
  color: var(--jp-ui-inverse-font-color1);
}

.jp-DirListing-item.lm-mod-drag-image,
.jp-DirListing-item.jp-mod-selected.lm-mod-drag-image {
  font-size: var(--jp-ui-font-size1);
  padding-left: 4px;
  margin-left: 4px;
  width: 160px;
  background-color: var(--jp-ui-inverse-font-color2);
  box-shadow: var(--jp-elevation-z2);
  border-radius: 0;
  color: var(--jp-ui-font-color1);
  transform: translateX(-40%) translateY(-58%);
}

.jp-Document {
  min-width: 120px;
  min-height: 120px;
  outline: none;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Main OutputArea
| OutputArea has a list of Outputs
|----------------------------------------------------------------------------*/

.jp-OutputArea {
  overflow-y: auto;
}

.jp-OutputArea-child {
  display: table;
  table-layout: fixed;
  width: 100%;
  overflow: hidden;
}

.jp-OutputPrompt {
  width: var(--jp-cell-prompt-width);
  color: var(--jp-cell-outprompt-font-color);
  font-family: var(--jp-cell-prompt-font-family);
  padding: var(--jp-code-padding);
  letter-spacing: var(--jp-cell-prompt-letter-spacing);
  line-height: var(--jp-code-line-height);
  font-size: var(--jp-code-font-size);
  border: var(--jp-border-width) solid transparent;
  opacity: var(--jp-cell-prompt-opacity);

  /* Right align prompt text, don't wrap to handle large prompt numbers */
  text-align: right;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;

  /* Disable text selection */
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
}

.jp-OutputArea-prompt {
  display: table-cell;
  vertical-align: top;
}

.jp-OutputArea-output {
  display: table-cell;
  width: 100%;
  height: auto;
  overflow: auto;
  user-select: text;
  -moz-user-select: text;
  -webkit-user-select: text;
  -ms-user-select: text;
}

.jp-OutputArea .jp-RenderedText {
  padding-left: 1ch;
}

/**
 * Prompt overlay.
 */

.jp-OutputArea-promptOverlay {
  position: absolute;
  top: 0;
  width: var(--jp-cell-prompt-width);
  height: 100%;
  opacity: 0.5;
}

.jp-OutputArea-promptOverlay:hover {
  background: var(--jp-layout-color2);
  box-shadow: inset 0 0 1px var(--jp-inverse-layout-color0);
  cursor: zoom-out;
}

.jp-mod-outputsScrolled .jp-OutputArea-promptOverlay:hover {
  cursor: zoom-in;
}

/**
 * Isolated output.
 */
.jp-OutputArea-output.jp-mod-isolated {
  width: 100%;
  display: block;
}

/*
When drag events occur, `lm-mod-override-cursor` is added to the body.
Because iframes steal all cursor events, the following two rules are necessary
to suppress pointer events while resize drags are occurring. There may be a
better solution to this problem.
*/
body.lm-mod-override-cursor .jp-OutputArea-output.jp-mod-isolated {
  position: relative;
}

body.lm-mod-override-cursor .jp-OutputArea-output.jp-mod-isolated::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: transparent;
}

/* pre */

.jp-OutputArea-output pre {
  border: none;
  margin: 0;
  padding: 0;
  overflow-x: auto;
  overflow-y: auto;
  word-break: break-all;
  word-wrap: break-word;
  white-space: pre-wrap;
}

/* tables */

.jp-OutputArea-output.jp-RenderedHTMLCommon table {
  margin-left: 0;
  margin-right: 0;
}

/* description lists */

.jp-OutputArea-output dl,
.jp-OutputArea-output dt,
.jp-OutputArea-output dd {
  display: block;
}

.jp-OutputArea-output dl {
  width: 100%;
  overflow: hidden;
  padding: 0;
  margin: 0;
}

.jp-OutputArea-output dt {
  font-weight: bold;
  float: left;
  width: 20%;
  padding: 0;
  margin: 0;
}

.jp-OutputArea-output dd {
  float: left;
  width: 80%;
  padding: 0;
  margin: 0;
}

.jp-TrimmedOutputs pre {
  background: var(--jp-layout-color3);
  font-size: calc(var(--jp-code-font-size) * 1.4);
  text-align: center;
  text-transform: uppercase;
}

/* Hide the gutter in case of
 *  - nested output areas (e.g. in the case of output widgets)
 *  - mirrored output areas
 */
.jp-OutputArea .jp-OutputArea .jp-OutputArea-prompt {
  display: none;
}

/* Hide empty lines in the output area, for instance due to cleared widgets */
.jp-OutputArea-prompt:empty {
  padding: 0;
  border: 0;
}

/*-----------------------------------------------------------------------------
| executeResult is added to any Output-result for the display of the object
| returned by a cell
|----------------------------------------------------------------------------*/

.jp-OutputArea-output.jp-OutputArea-executeResult {
  margin-left: 0;
  width: 100%;
}

/* Text output with the Out[] prompt needs a top padding to match the
 * alignment of the Out[] prompt itself.
 */
.jp-OutputArea-executeResult .jp-RenderedText.jp-OutputArea-output {
  padding-top: var(--jp-code-padding);
  border-top: var(--jp-border-width) solid transparent;
}

/*-----------------------------------------------------------------------------
| The Stdin output
|----------------------------------------------------------------------------*/

.jp-Stdin-prompt {
  color: var(--jp-content-font-color0);
  padding-right: var(--jp-code-padding);
  vertical-align: baseline;
  flex: 0 0 auto;
}

.jp-Stdin-input {
  font-family: var(--jp-code-font-family);
  font-size: inherit;
  color: inherit;
  background-color: inherit;
  width: 42%;
  min-width: 200px;

  /* make sure input baseline aligns with prompt */
  vertical-align: baseline;

  /* padding + margin = 0.5em between prompt and cursor */
  padding: 0 0.25em;
  margin: 0 0.25em;
  flex: 0 0 70%;
}

.jp-Stdin-input::placeholder {
  opacity: 0;
}

.jp-Stdin-input:focus {
  box-shadow: none;
}

.jp-Stdin-input:focus::placeholder {
  opacity: 1;
}

/*-----------------------------------------------------------------------------
| Output Area View
|----------------------------------------------------------------------------*/

.jp-LinkedOutputView .jp-OutputArea {
  height: 100%;
  display: block;
}

.jp-LinkedOutputView .jp-OutputArea-output:only-child {
  height: 100%;
}

/*-----------------------------------------------------------------------------
| Printing
|----------------------------------------------------------------------------*/

@media print {
  .jp-OutputArea-child {
    break-inside: avoid-page;
  }
}

/*-----------------------------------------------------------------------------
| Mobile
|----------------------------------------------------------------------------*/
@media only screen and (max-width: 760px) {
  .jp-OutputPrompt {
    display: table-row;
    text-align: left;
  }

  .jp-OutputArea-child .jp-OutputArea-output {
    display: table-row;
    margin-left: var(--jp-notebook-padding);
  }
}

/* Trimmed outputs warning */
.jp-TrimmedOutputs > a {
  margin: 10px;
  text-decoration: none;
  cursor: pointer;
}

.jp-TrimmedOutputs > a:hover {
  text-decoration: none;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Table of Contents
|----------------------------------------------------------------------------*/

:root {
  --jp-private-toc-active-width: 4px;
}

.jp-TableOfContents {
  display: flex;
  flex-direction: column;
  background: var(--jp-layout-color1);
  color: var(--jp-ui-font-color1);
  font-size: var(--jp-ui-font-size1);
  height: 100%;
}

.jp-TableOfContents-placeholder {
  text-align: center;
}

.jp-TableOfContents-placeholderContent {
  color: var(--jp-content-font-color2);
  padding: 8px;
}

.jp-TableOfContents-placeholderContent > h3 {
  margin-bottom: var(--jp-content-heading-margin-bottom);
}

.jp-TableOfContents .jp-SidePanel-content {
  overflow-y: auto;
}

.jp-TableOfContents-tree {
  margin: 4px;
}

.jp-TableOfContents ol {
  list-style-type: none;
}

/* stylelint-disable-next-line selector-max-type */
.jp-TableOfContents li > ol {
  /* Align left border with triangle icon center */
  padding-left: 11px;
}

.jp-TableOfContents-content {
  /* left margin for the active heading indicator */
  margin: 0 0 0 var(--jp-private-toc-active-width);
  padding: 0;
  background-color: var(--jp-layout-color1);
}

.jp-tocItem {
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
}

.jp-tocItem-heading {
  display: flex;
  cursor: pointer;
}

.jp-tocItem-heading:hover {
  background-color: var(--jp-layout-color2);
}

.jp-tocItem-content {
  display: block;
  padding: 4px 0;
  white-space: nowrap;
  text-overflow: ellipsis;
  overflow-x: hidden;
}

.jp-tocItem-collapser {
  height: 20px;
  margin: 2px 2px 0;
  padding: 0;
  background: none;
  border: none;
  cursor: pointer;
}

.jp-tocItem-collapser:hover {
  background-color: var(--jp-layout-color3);
}

/* Active heading indicator */

.jp-tocItem-heading::before {
  content: ' ';
  background: transparent;
  width: var(--jp-private-toc-active-width);
  height: 24px;
  position: absolute;
  left: 0;
  border-radius: var(--jp-border-radius);
}

.jp-tocItem-heading.jp-tocItem-active::before {
  background-color: var(--jp-brand-color1);
}

.jp-tocItem-heading:hover.jp-tocItem-active::before {
  background: var(--jp-brand-color0);
  opacity: 1;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-Collapser {
  flex: 0 0 var(--jp-cell-collapser-width);
  padding: 0;
  margin: 0;
  border: none;
  outline: none;
  background: transparent;
  border-radius: var(--jp-border-radius);
  opacity: 1;
}

.jp-Collapser-child {
  display: block;
  width: 100%;
  box-sizing: border-box;

  /* height: 100% doesn't work because the height of its parent is computed from content */
  position: absolute;
  top: 0;
  bottom: 0;
}

/*-----------------------------------------------------------------------------
| Printing
|----------------------------------------------------------------------------*/

/*
Hiding collapsers in print mode.

Note: input and output wrappers have "display: block" propery in print mode.
*/

@media print {
  .jp-Collapser {
    display: none;
  }
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Header/Footer
|----------------------------------------------------------------------------*/

/* Hidden by zero height by default */
.jp-CellHeader,
.jp-CellFooter {
  height: 0;
  width: 100%;
  padding: 0;
  margin: 0;
  border: none;
  outline: none;
  background: transparent;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Input
|----------------------------------------------------------------------------*/

/* All input areas */
.jp-InputArea {
  display: table;
  table-layout: fixed;
  width: 100%;
  overflow: hidden;
}

.jp-InputArea-editor {
  display: table-cell;
  overflow: hidden;
  vertical-align: top;

  /* This is the non-active, default styling */
  border: var(--jp-border-width) solid var(--jp-cell-editor-border-color);
  border-radius: 0;
  background: var(--jp-cell-editor-background);
}

.jp-InputPrompt {
  display: table-cell;
  vertical-align: top;
  width: var(--jp-cell-prompt-width);
  color: var(--jp-cell-inprompt-font-color);
  font-family: var(--jp-cell-prompt-font-family);
  padding: var(--jp-code-padding);
  letter-spacing: var(--jp-cell-prompt-letter-spacing);
  opacity: var(--jp-cell-prompt-opacity);
  line-height: var(--jp-code-line-height);
  font-size: var(--jp-code-font-size);
  border: var(--jp-border-width) solid transparent;

  /* Right align prompt text, don't wrap to handle large prompt numbers */
  text-align: right;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;

  /* Disable text selection */
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
}

/*-----------------------------------------------------------------------------
| Mobile
|----------------------------------------------------------------------------*/
@media only screen and (max-width: 760px) {
  .jp-InputArea-editor {
    display: table-row;
    margin-left: var(--jp-notebook-padding);
  }

  .jp-InputPrompt {
    display: table-row;
    text-align: left;
  }
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Placeholder
|----------------------------------------------------------------------------*/

.jp-Placeholder {
  display: table;
  table-layout: fixed;
  width: 100%;
}

.jp-Placeholder-prompt {
  display: table-cell;
  box-sizing: border-box;
}

.jp-Placeholder-content {
  display: table-cell;
  padding: 4px 6px;
  border: 1px solid transparent;
  border-radius: 0;
  background: none;
  box-sizing: border-box;
  cursor: pointer;
}

.jp-Placeholder-contentContainer {
  display: flex;
}

.jp-Placeholder-content:hover,
.jp-InputPlaceholder > .jp-Placeholder-content:hover {
  border-color: var(--jp-layout-color3);
}

.jp-Placeholder-content .jp-MoreHorizIcon {
  width: 32px;
  height: 16px;
  border: 1px solid transparent;
  border-radius: var(--jp-border-radius);
}

.jp-Placeholder-content .jp-MoreHorizIcon:hover {
  border: 1px solid var(--jp-border-color1);
  box-shadow: 0 0 2px 0 rgba(0, 0, 0, 0.25);
  background-color: var(--jp-layout-color0);
}

.jp-PlaceholderText {
  white-space: nowrap;
  overflow-x: hidden;
  color: var(--jp-inverse-layout-color3);
  font-family: var(--jp-code-font-family);
}

.jp-InputPlaceholder > .jp-Placeholder-content {
  border-color: var(--jp-cell-editor-border-color);
  background: var(--jp-cell-editor-background);
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Private CSS variables
|----------------------------------------------------------------------------*/

:root {
  --jp-private-cell-scrolling-output-offset: 5px;
}

/*-----------------------------------------------------------------------------
| Cell
|----------------------------------------------------------------------------*/

.jp-Cell {
  padding: var(--jp-cell-padding);
  margin: 0;
  border: none;
  outline: none;
  background: transparent;
}

/*-----------------------------------------------------------------------------
| Common input/output
|----------------------------------------------------------------------------*/

.jp-Cell-inputWrapper,
.jp-Cell-outputWrapper {
  display: flex;
  flex-direction: row;
  padding: 0;
  margin: 0;

  /* Added to reveal the box-shadow on the input and output collapsers. */
  overflow: visible;
}

/* Only input/output areas inside cells */
.jp-Cell-inputArea,
.jp-Cell-outputArea {
  flex: 1 1 auto;
}

/*-----------------------------------------------------------------------------
| Collapser
|----------------------------------------------------------------------------*/

/* Make the output collapser disappear when there is not output, but do so
 * in a manner that leaves it in the layout and preserves its width.
 */
.jp-Cell.jp-mod-noOutputs .jp-Cell-outputCollapser {
  border: none !important;
  background: transparent !important;
}

.jp-Cell:not(.jp-mod-noOutputs) .jp-Cell-outputCollapser {
  min-height: var(--jp-cell-collapser-min-height);
}

/*-----------------------------------------------------------------------------
| Output
|----------------------------------------------------------------------------*/

/* Put a space between input and output when there IS output */
.jp-Cell:not(.jp-mod-noOutputs) .jp-Cell-outputWrapper {
  margin-top: 5px;
}

.jp-CodeCell.jp-mod-outputsScrolled .jp-Cell-outputArea {
  overflow-y: auto;
  max-height: 24em;
  margin-left: var(--jp-private-cell-scrolling-output-offset);
  resize: vertical;
}

.jp-CodeCell.jp-mod-outputsScrolled .jp-Cell-outputArea[style*='height'] {
  max-height: unset;
}

.jp-CodeCell.jp-mod-outputsScrolled .jp-Cell-outputArea::after {
  content: ' ';
  box-shadow: inset 0 0 6px 2px rgb(0 0 0 / 30%);
  width: 100%;
  height: 100%;
  position: sticky;
  bottom: 0;
  top: 0;
  margin-top: -50%;
  float: left;
  display: block;
  pointer-events: none;
}

.jp-CodeCell.jp-mod-outputsScrolled .jp-OutputArea-child {
  padding-top: 6px;
}

.jp-CodeCell.jp-mod-outputsScrolled .jp-OutputArea-prompt {
  width: calc(
    var(--jp-cell-prompt-width) - var(--jp-private-cell-scrolling-output-offset)
  );
}

.jp-CodeCell.jp-mod-outputsScrolled .jp-OutputArea-promptOverlay {
  left: calc(-1 * var(--jp-private-cell-scrolling-output-offset));
}

/*-----------------------------------------------------------------------------
| CodeCell
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| MarkdownCell
|----------------------------------------------------------------------------*/

.jp-MarkdownOutput {
  display: table-cell;
  width: 100%;
  margin-top: 0;
  margin-bottom: 0;
  padding-left: var(--jp-code-padding);
}

.jp-MarkdownOutput.jp-RenderedHTMLCommon {
  overflow: auto;
}

/* collapseHeadingButton (show always if hiddenCellsButton is _not_ shown) */
.jp-collapseHeadingButton {
  display: flex;
  min-height: var(--jp-cell-collapser-min-height);
  font-size: var(--jp-code-font-size);
  position: absolute;
  background-color: transparent;
  background-size: 25px;
  background-repeat: no-repeat;
  background-position-x: center;
  background-position-y: top;
  background-image: var(--jp-icon-caret-down);
  right: 0;
  top: 0;
  bottom: 0;
}

.jp-collapseHeadingButton.jp-mod-collapsed {
  background-image: var(--jp-icon-caret-right);
}

/*
 set the container font size to match that of content
 so that the nested collapse buttons have the right size
*/
.jp-MarkdownCell .jp-InputPrompt {
  font-size: var(--jp-content-font-size1);
}

/*
  Align collapseHeadingButton with cell top header
  The font sizes are identical to the ones in packages/rendermime/style/base.css
*/
.jp-mod-rendered .jp-collapseHeadingButton[data-heading-level='1'] {
  font-size: var(--jp-content-font-size5);
  background-position-y: calc(0.3 * var(--jp-content-font-size5));
}

.jp-mod-rendered .jp-collapseHeadingButton[data-heading-level='2'] {
  font-size: var(--jp-content-font-size4);
  background-position-y: calc(0.3 * var(--jp-content-font-size4));
}

.jp-mod-rendered .jp-collapseHeadingButton[data-heading-level='3'] {
  font-size: var(--jp-content-font-size3);
  background-position-y: calc(0.3 * var(--jp-content-font-size3));
}

.jp-mod-rendered .jp-collapseHeadingButton[data-heading-level='4'] {
  font-size: var(--jp-content-font-size2);
  background-position-y: calc(0.3 * var(--jp-content-font-size2));
}

.jp-mod-rendered .jp-collapseHeadingButton[data-heading-level='5'] {
  font-size: var(--jp-content-font-size1);
  background-position-y: top;
}

.jp-mod-rendered .jp-collapseHeadingButton[data-heading-level='6'] {
  font-size: var(--jp-content-font-size0);
  background-position-y: top;
}

/* collapseHeadingButton (show only on (hover,active) if hiddenCellsButton is shown) */
.jp-Notebook.jp-mod-showHiddenCellsButton .jp-collapseHeadingButton {
  display: none;
}

.jp-Notebook.jp-mod-showHiddenCellsButton
  :is(.jp-MarkdownCell:hover, .jp-mod-active)
  .jp-collapseHeadingButton {
  display: flex;
}

/* showHiddenCellsButton (only show if jp-mod-showHiddenCellsButton is set, which
is a consequence of the showHiddenCellsButton option in Notebook Settings)*/
.jp-Notebook.jp-mod-showHiddenCellsButton .jp-showHiddenCellsButton {
  margin-left: calc(var(--jp-cell-prompt-width) + 2 * var(--jp-code-padding));
  margin-top: var(--jp-code-padding);
  border: 1px solid var(--jp-border-color2);
  background-color: var(--jp-border-color3) !important;
  color: var(--jp-content-font-color0) !important;
  display: flex;
}

.jp-Notebook.jp-mod-showHiddenCellsButton .jp-showHiddenCellsButton:hover {
  background-color: var(--jp-border-color2) !important;
}

.jp-showHiddenCellsButton {
  display: none;
}

/*-----------------------------------------------------------------------------
| Printing
|----------------------------------------------------------------------------*/

/*
Using block instead of flex to allow the use of the break-inside CSS property for
cell outputs.
*/

@media print {
  .jp-Cell-inputWrapper,
  .jp-Cell-outputWrapper {
    display: block;
  }
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Variables
|----------------------------------------------------------------------------*/

:root {
  --jp-notebook-toolbar-padding: 2px 5px 2px 2px;
}

/*-----------------------------------------------------------------------------

/*-----------------------------------------------------------------------------
| Styles
|----------------------------------------------------------------------------*/

.jp-NotebookPanel-toolbar {
  padding: var(--jp-notebook-toolbar-padding);

  /* disable paint containment from lumino 2.0 default strict CSS containment */
  contain: style size !important;
}

.jp-Toolbar-item.jp-Notebook-toolbarCellType .jp-select-wrapper.jp-mod-focused {
  border: none;
  box-shadow: none;
}

.jp-Notebook-toolbarCellTypeDropdown select {
  height: 24px;
  font-size: var(--jp-ui-font-size1);
  line-height: 14px;
  border-radius: 0;
  display: block;
}

.jp-Notebook-toolbarCellTypeDropdown span {
  top: 5px !important;
}

.jp-Toolbar-responsive-popup {
  position: absolute;
  height: fit-content;
  display: flex;
  flex-direction: row;
  flex-wrap: wrap;
  justify-content: flex-end;
  border-bottom: var(--jp-border-width) solid var(--jp-toolbar-border-color);
  box-shadow: var(--jp-toolbar-box-shadow);
  background: var(--jp-toolbar-background);
  min-height: var(--jp-toolbar-micro-height);
  padding: var(--jp-notebook-toolbar-padding);
  z-index: 1;
  right: 0;
  top: 0;
}

.jp-Toolbar > .jp-Toolbar-responsive-opener {
  margin-left: auto;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Variables
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------

/*-----------------------------------------------------------------------------
| Styles
|----------------------------------------------------------------------------*/

.jp-Notebook-ExecutionIndicator {
  position: relative;
  display: inline-block;
  height: 100%;
  z-index: 9997;
}

.jp-Notebook-ExecutionIndicator-tooltip {
  visibility: hidden;
  height: auto;
  width: max-content;
  width: -moz-max-content;
  background-color: var(--jp-layout-color2);
  color: var(--jp-ui-font-color1);
  text-align: justify;
  border-radius: 6px;
  padding: 0 5px;
  position: fixed;
  display: table;
}

.jp-Notebook-ExecutionIndicator-tooltip.up {
  transform: translateX(-50%) translateY(-100%) translateY(-32px);
}

.jp-Notebook-ExecutionIndicator-tooltip.down {
  transform: translateX(calc(-100% + 16px)) translateY(5px);
}

.jp-Notebook-ExecutionIndicator-tooltip.hidden {
  display: none;
}

.jp-Notebook-ExecutionIndicator:hover .jp-Notebook-ExecutionIndicator-tooltip {
  visibility: visible;
}

.jp-Notebook-ExecutionIndicator span {
  font-size: var(--jp-ui-font-size1);
  font-family: var(--jp-ui-font-family);
  color: var(--jp-ui-font-color1);
  line-height: 24px;
  display: block;
}

.jp-Notebook-ExecutionIndicator-progress-bar {
  display: flex;
  justify-content: center;
  height: 100%;
}

/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

/*
 * Execution indicator
 */
.jp-tocItem-content::after {
  content: '';

  /* Must be identical to form a circle */
  width: 12px;
  height: 12px;
  background: none;
  border: none;
  position: absolute;
  right: 0;
}

.jp-tocItem-content[data-running='0']::after {
  border-radius: 50%;
  border: var(--jp-border-width) solid var(--jp-inverse-layout-color3);
  background: none;
}

.jp-tocItem-content[data-running='1']::after {
  border-radius: 50%;
  border: var(--jp-border-width) solid var(--jp-inverse-layout-color3);
  background-color: var(--jp-inverse-layout-color3);
}

.jp-tocItem-content[data-running='0'],
.jp-tocItem-content[data-running='1'] {
  margin-right: 12px;
}

/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

.jp-Notebook-footer {
  height: 27px;
  margin-left: calc(
    var(--jp-cell-prompt-width) + var(--jp-cell-collapser-width) +
      var(--jp-cell-padding)
  );
  width: calc(
    100% -
      (
        var(--jp-cell-prompt-width) + var(--jp-cell-collapser-width) +
          var(--jp-cell-padding) + var(--jp-cell-padding)
      )
  );
  border: var(--jp-border-width) solid var(--jp-cell-editor-border-color);
  color: var(--jp-ui-font-color3);
  margin-top: 6px;
  background: none;
  cursor: pointer;
}

.jp-Notebook-footer:focus {
  border-color: var(--jp-cell-editor-active-border-color);
}

/* For devices that support hovering, hide footer until hover */
@media (hover: hover) {
  .jp-Notebook-footer {
    opacity: 0;
  }

  .jp-Notebook-footer:focus,
  .jp-Notebook-footer:hover {
    opacity: 1;
  }
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Imports
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| CSS variables
|----------------------------------------------------------------------------*/

:root {
  --jp-side-by-side-output-size: 1fr;
  --jp-side-by-side-resized-cell: var(--jp-side-by-side-output-size);
  --jp-private-notebook-dragImage-width: 304px;
  --jp-private-notebook-dragImage-height: 36px;
  --jp-private-notebook-selected-color: var(--md-blue-400);
  --jp-private-notebook-active-color: var(--md-green-400);
}

/*-----------------------------------------------------------------------------
| Notebook
|----------------------------------------------------------------------------*/

/* stylelint-disable selector-max-class */

.jp-NotebookPanel {
  display: block;
  height: 100%;
}

.jp-NotebookPanel.jp-Document {
  min-width: 240px;
  min-height: 120px;
}

.jp-Notebook {
  padding: var(--jp-notebook-padding);
  outline: none;
  overflow: auto;
  background: var(--jp-layout-color0);
}

.jp-Notebook.jp-mod-scrollPastEnd::after {
  display: block;
  content: '';
  min-height: var(--jp-notebook-scroll-padding);
}

.jp-MainAreaWidget-ContainStrict .jp-Notebook * {
  contain: strict;
}

.jp-Notebook .jp-Cell {
  overflow: visible;
}

.jp-Notebook .jp-Cell .jp-InputPrompt {
  cursor: move;
}

/*-----------------------------------------------------------------------------
| Notebook state related styling
|
| The notebook and cells each have states, here are the possibilities:
|
| - Notebook
|   - Command
|   - Edit
| - Cell
|   - None
|   - Active (only one can be active)
|   - Selected (the cells actions are applied to)
|   - Multiselected (when multiple selected, the cursor)
|   - No outputs
|----------------------------------------------------------------------------*/

/* Command or edit modes */

.jp-Notebook .jp-Cell:not(.jp-mod-active) .jp-InputPrompt {
  opacity: var(--jp-cell-prompt-not-active-opacity);
  color: var(--jp-cell-prompt-not-active-font-color);
}

.jp-Notebook .jp-Cell:not(.jp-mod-active) .jp-OutputPrompt {
  opacity: var(--jp-cell-prompt-not-active-opacity);
  color: var(--jp-cell-prompt-not-active-font-color);
}

/* cell is active */
.jp-Notebook .jp-Cell.jp-mod-active .jp-Collapser {
  background: var(--jp-brand-color1);
}

/* cell is dirty */
.jp-Notebook .jp-Cell.jp-mod-dirty .jp-InputPrompt {
  color: var(--jp-warn-color1);
}

.jp-Notebook .jp-Cell.jp-mod-dirty .jp-InputPrompt::before {
  color: var(--jp-warn-color1);
  content: '•';
}

.jp-Notebook .jp-Cell.jp-mod-active.jp-mod-dirty .jp-Collapser {
  background: var(--jp-warn-color1);
}

/* collapser is hovered */
.jp-Notebook .jp-Cell .jp-Collapser:hover {
  box-shadow: var(--jp-elevation-z2);
  background: var(--jp-brand-color1);
  opacity: var(--jp-cell-collapser-not-active-hover-opacity);
}

/* cell is active and collapser is hovered */
.jp-Notebook .jp-Cell.jp-mod-active .jp-Collapser:hover {
  background: var(--jp-brand-color0);
  opacity: 1;
}

/* Command mode */

.jp-Notebook.jp-mod-commandMode .jp-Cell.jp-mod-selected {
  background: var(--jp-notebook-multiselected-color);
}

.jp-Notebook.jp-mod-commandMode
  .jp-Cell.jp-mod-active.jp-mod-selected:not(.jp-mod-multiSelected) {
  background: transparent;
}

/* Edit mode */

.jp-Notebook.jp-mod-editMode .jp-Cell.jp-mod-active .jp-InputArea-editor {
  border: var(--jp-border-width) solid var(--jp-cell-editor-active-border-color);
  box-shadow: var(--jp-input-box-shadow);
  background-color: var(--jp-cell-editor-active-background);
}

/*-----------------------------------------------------------------------------
| Notebook drag and drop
|----------------------------------------------------------------------------*/

.jp-Notebook-cell.jp-mod-dropSource {
  opacity: 0.5;
}

.jp-Notebook-cell.jp-mod-dropTarget,
.jp-Notebook.jp-mod-commandMode
  .jp-Notebook-cell.jp-mod-active.jp-mod-selected.jp-mod-dropTarget {
  border-top-color: var(--jp-private-notebook-selected-color);
  border-top-style: solid;
  border-top-width: 2px;
}

.jp-dragImage {
  display: block;
  flex-direction: row;
  width: var(--jp-private-notebook-dragImage-width);
  height: var(--jp-private-notebook-dragImage-height);
  border: var(--jp-border-width) solid var(--jp-cell-editor-border-color);
  background: var(--jp-cell-editor-background);
  overflow: visible;
}

.jp-dragImage-singlePrompt {
  box-shadow: 2px 2px 4px 0 rgba(0, 0, 0, 0.12);
}

.jp-dragImage .jp-dragImage-content {
  flex: 1 1 auto;
  z-index: 2;
  font-size: var(--jp-code-font-size);
  font-family: var(--jp-code-font-family);
  line-height: var(--jp-code-line-height);
  padding: var(--jp-code-padding);
  border: var(--jp-border-width) solid var(--jp-cell-editor-border-color);
  background: var(--jp-cell-editor-background-color);
  color: var(--jp-content-font-color3);
  text-align: left;
  margin: 4px 4px 4px 0;
}

.jp-dragImage .jp-dragImage-prompt {
  flex: 0 0 auto;
  min-width: 36px;
  color: var(--jp-cell-inprompt-font-color);
  padding: var(--jp-code-padding);
  padding-left: 12px;
  font-family: var(--jp-cell-prompt-font-family);
  letter-spacing: var(--jp-cell-prompt-letter-spacing);
  line-height: 1.9;
  font-size: var(--jp-code-font-size);
  border: var(--jp-border-width) solid transparent;
}

.jp-dragImage-multipleBack {
  z-index: -1;
  position: absolute;
  height: 32px;
  width: 300px;
  top: 8px;
  left: 8px;
  background: var(--jp-layout-color2);
  border: var(--jp-border-width) solid var(--jp-input-border-color);
  box-shadow: 2px 2px 4px 0 rgba(0, 0, 0, 0.12);
}

/*-----------------------------------------------------------------------------
| Cell toolbar
|----------------------------------------------------------------------------*/

.jp-NotebookTools {
  display: block;
  min-width: var(--jp-sidebar-min-width);
  color: var(--jp-ui-font-color1);
  background: var(--jp-layout-color1);

  /* This is needed so that all font sizing of children done in ems is
    * relative to this base size */
  font-size: var(--jp-ui-font-size1);
  overflow: auto;
}

.jp-ActiveCellTool {
  padding: 12px 0;
  display: flex;
}

.jp-ActiveCellTool-Content {
  flex: 1 1 auto;
}

.jp-ActiveCellTool .jp-ActiveCellTool-CellContent {
  background: var(--jp-cell-editor-background);
  border: var(--jp-border-width) solid var(--jp-cell-editor-border-color);
  border-radius: 0;
  min-height: 29px;
}

.jp-ActiveCellTool .jp-InputPrompt {
  min-width: calc(var(--jp-cell-prompt-width) * 0.75);
}

.jp-ActiveCellTool-CellContent > pre {
  padding: 5px 4px;
  margin: 0;
  white-space: normal;
}

.jp-MetadataEditorTool {
  flex-direction: column;
  padding: 12px 0;
}

.jp-RankedPanel > :not(:first-child) {
  margin-top: 12px;
}

.jp-KeySelector select.jp-mod-styled {
  font-size: var(--jp-ui-font-size1);
  color: var(--jp-ui-font-color0);
  border: var(--jp-border-width) solid var(--jp-border-color1);
}

.jp-KeySelector label,
.jp-MetadataEditorTool label,
.jp-NumberSetter label {
  line-height: 1.4;
}

.jp-NotebookTools .jp-select-wrapper {
  margin-top: 4px;
  margin-bottom: 0;
}

.jp-NumberSetter input {
  width: 100%;
  margin-top: 4px;
}

.jp-NotebookTools .jp-Collapse {
  margin-top: 16px;
}

/*-----------------------------------------------------------------------------
| Presentation Mode (.jp-mod-presentationMode)
|----------------------------------------------------------------------------*/

.jp-mod-presentationMode .jp-Notebook {
  --jp-content-font-size1: var(--jp-content-presentation-font-size1);
  --jp-code-font-size: var(--jp-code-presentation-font-size);
}

.jp-mod-presentationMode .jp-Notebook .jp-Cell .jp-InputPrompt,
.jp-mod-presentationMode .jp-Notebook .jp-Cell .jp-OutputPrompt {
  flex: 0 0 110px;
}

/*-----------------------------------------------------------------------------
| Side-by-side Mode (.jp-mod-sideBySide)
|----------------------------------------------------------------------------*/
.jp-mod-sideBySide.jp-Notebook .jp-Notebook-cell {
  margin-top: 3em;
  margin-bottom: 3em;
  margin-left: 5%;
  margin-right: 5%;
}

.jp-mod-sideBySide.jp-Notebook .jp-CodeCell {
  display: grid;
  grid-template-columns: minmax(0, 1fr) min-content minmax(
      0,
      var(--jp-side-by-side-output-size)
    );
  grid-template-rows: auto minmax(0, 1fr) auto;
  grid-template-areas:
    'header header header'
    'input handle output'
    'footer footer footer';
}

.jp-mod-sideBySide.jp-Notebook .jp-CodeCell.jp-mod-resizedCell {
  grid-template-columns: minmax(0, 1fr) min-content minmax(
      0,
      var(--jp-side-by-side-resized-cell)
    );
}

.jp-mod-sideBySide.jp-Notebook .jp-CodeCell .jp-CellHeader {
  grid-area: header;
}

.jp-mod-sideBySide.jp-Notebook .jp-CodeCell .jp-Cell-inputWrapper {
  grid-area: input;
}

.jp-mod-sideBySide.jp-Notebook .jp-CodeCell .jp-Cell-outputWrapper {
  /* overwrite the default margin (no vertical separation needed in side by side move */
  margin-top: 0;
  grid-area: output;
}

.jp-mod-sideBySide.jp-Notebook .jp-CodeCell .jp-CellFooter {
  grid-area: footer;
}

.jp-mod-sideBySide.jp-Notebook .jp-CodeCell .jp-CellResizeHandle {
  grid-area: handle;
  user-select: none;
  display: block;
  height: 100%;
  cursor: ew-resize;
  padding: 0 var(--jp-cell-padding);
}

.jp-mod-sideBySide.jp-Notebook .jp-CodeCell .jp-CellResizeHandle::after {
  content: '';
  display: block;
  background: var(--jp-border-color2);
  height: 100%;
  width: 5px;
}

.jp-mod-sideBySide.jp-Notebook
  .jp-CodeCell.jp-mod-resizedCell
  .jp-CellResizeHandle::after {
  background: var(--jp-border-color0);
}

.jp-CellResizeHandle {
  display: none;
}

/*-----------------------------------------------------------------------------
| Placeholder
|----------------------------------------------------------------------------*/

.jp-Cell-Placeholder {
  padding-left: 55px;
}

.jp-Cell-Placeholder-wrapper {
  background: #fff;
  border: 1px solid;
  border-color: #e5e6e9 #dfe0e4 #d0d1d5;
  border-radius: 4px;
  -webkit-border-radius: 4px;
  margin: 10px 15px;
}

.jp-Cell-Placeholder-wrapper-inner {
  padding: 15px;
  position: relative;
}

.jp-Cell-Placeholder-wrapper-body {
  background-repeat: repeat;
  background-size: 50% auto;
}

.jp-Cell-Placeholder-wrapper-body div {
  background: #f6f7f8;
  background-image: -webkit-linear-gradient(
    left,
    #f6f7f8 0%,
    #edeef1 20%,
    #f6f7f8 40%,
    #f6f7f8 100%
  );
  background-repeat: no-repeat;
  background-size: 800px 104px;
  height: 104px;
  position: absolute;
  right: 15px;
  left: 15px;
  top: 15px;
}

div.jp-Cell-Placeholder-h1 {
  top: 20px;
  height: 20px;
  left: 15px;
  width: 150px;
}

div.jp-Cell-Placeholder-h2 {
  left: 15px;
  top: 50px;
  height: 10px;
  width: 100px;
}

div.jp-Cell-Placeholder-content-1,
div.jp-Cell-Placeholder-content-2,
div.jp-Cell-Placeholder-content-3 {
  left: 15px;
  right: 15px;
  height: 10px;
}

div.jp-Cell-Placeholder-content-1 {
  top: 100px;
}

div.jp-Cell-Placeholder-content-2 {
  top: 120px;
}

div.jp-Cell-Placeholder-content-3 {
  top: 140px;
}

</style>
<style type="text/css">
/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*
The following CSS variables define the main, public API for styling JupyterLab.
These variables should be used by all plugins wherever possible. In other
words, plugins should not define custom colors, sizes, etc unless absolutely
necessary. This enables users to change the visual theme of JupyterLab
by changing these variables.

Many variables appear in an ordered sequence (0,1,2,3). These sequences
are designed to work well together, so for example, `--jp-border-color1` should
be used with `--jp-layout-color1`. The numbers have the following meanings:

* 0: super-primary, reserved for special emphasis
* 1: primary, most important under normal situations
* 2: secondary, next most important under normal situations
* 3: tertiary, next most important under normal situations

Throughout JupyterLab, we are mostly following principles from Google's
Material Design when selecting colors. We are not, however, following
all of MD as it is not optimized for dense, information rich UIs.
*/

:root {
  /* Elevation
   *
   * We style box-shadows using Material Design's idea of elevation. These particular numbers are taken from here:
   *
   * https://github.com/material-components/material-components-web
   * https://material-components-web.appspot.com/elevation.html
   */

  --jp-shadow-base-lightness: 0;
  --jp-shadow-umbra-color: rgba(
    var(--jp-shadow-base-lightness),
    var(--jp-shadow-base-lightness),
    var(--jp-shadow-base-lightness),
    0.2
  );
  --jp-shadow-penumbra-color: rgba(
    var(--jp-shadow-base-lightness),
    var(--jp-shadow-base-lightness),
    var(--jp-shadow-base-lightness),
    0.14
  );
  --jp-shadow-ambient-color: rgba(
    var(--jp-shadow-base-lightness),
    var(--jp-shadow-base-lightness),
    var(--jp-shadow-base-lightness),
    0.12
  );
  --jp-elevation-z0: none;
  --jp-elevation-z1: 0 2px 1px -1px var(--jp-shadow-umbra-color),
    0 1px 1px 0 var(--jp-shadow-penumbra-color),
    0 1px 3px 0 var(--jp-shadow-ambient-color);
  --jp-elevation-z2: 0 3px 1px -2px var(--jp-shadow-umbra-color),
    0 2px 2px 0 var(--jp-shadow-penumbra-color),
    0 1px 5px 0 var(--jp-shadow-ambient-color);
  --jp-elevation-z4: 0 2px 4px -1px var(--jp-shadow-umbra-color),
    0 4px 5px 0 var(--jp-shadow-penumbra-color),
    0 1px 10px 0 var(--jp-shadow-ambient-color);
  --jp-elevation-z6: 0 3px 5px -1px var(--jp-shadow-umbra-color),
    0 6px 10px 0 var(--jp-shadow-penumbra-color),
    0 1px 18px 0 var(--jp-shadow-ambient-color);
  --jp-elevation-z8: 0 5px 5px -3px var(--jp-shadow-umbra-color),
    0 8px 10px 1px var(--jp-shadow-penumbra-color),
    0 3px 14px 2px var(--jp-shadow-ambient-color);
  --jp-elevation-z12: 0 7px 8px -4px var(--jp-shadow-umbra-color),
    0 12px 17px 2px var(--jp-shadow-penumbra-color),
    0 5px 22px 4px var(--jp-shadow-ambient-color);
  --jp-elevation-z16: 0 8px 10px -5px var(--jp-shadow-umbra-color),
    0 16px 24px 2px var(--jp-shadow-penumbra-color),
    0 6px 30px 5px var(--jp-shadow-ambient-color);
  --jp-elevation-z20: 0 10px 13px -6px var(--jp-shadow-umbra-color),
    0 20px 31px 3px var(--jp-shadow-penumbra-color),
    0 8px 38px 7px var(--jp-shadow-ambient-color);
  --jp-elevation-z24: 0 11px 15px -7px var(--jp-shadow-umbra-color),
    0 24px 38px 3px var(--jp-shadow-penumbra-color),
    0 9px 46px 8px var(--jp-shadow-ambient-color);

  /* Borders
   *
   * The following variables, specify the visual styling of borders in JupyterLab.
   */

  --jp-border-width: 1px;
  --jp-border-color0: var(--md-grey-400);
  --jp-border-color1: var(--md-grey-400);
  --jp-border-color2: var(--md-grey-300);
  --jp-border-color3: var(--md-grey-200);
  --jp-inverse-border-color: var(--md-grey-600);
  --jp-border-radius: 2px;

  /* UI Fonts
   *
   * The UI font CSS variables are used for the typography all of the JupyterLab
   * user interface elements that are not directly user generated content.
   *
   * The font sizing here is done assuming that the body font size of --jp-ui-font-size1
   * is applied to a parent element. When children elements, such as headings, are sized
   * in em all things will be computed relative to that body size.
   */

  --jp-ui-font-scale-factor: 1.2;
  --jp-ui-font-size0: 0.83333em;
  --jp-ui-font-size1: 13px; /* Base font size */
  --jp-ui-font-size2: 1.2em;
  --jp-ui-font-size3: 1.44em;
  --jp-ui-font-family: system-ui, -apple-system, blinkmacsystemfont, 'Segoe UI',
    helvetica, arial, sans-serif, 'Apple Color Emoji', 'Segoe UI Emoji',
    'Segoe UI Symbol';

  /*
   * Use these font colors against the corresponding main layout colors.
   * In a light theme, these go from dark to light.
   */

  /* Defaults use Material Design specification */
  --jp-ui-font-color0: rgba(0, 0, 0, 1);
  --jp-ui-font-color1: rgba(0, 0, 0, 0.87);
  --jp-ui-font-color2: rgba(0, 0, 0, 0.54);
  --jp-ui-font-color3: rgba(0, 0, 0, 0.38);

  /*
   * Use these against the brand/accent/warn/error colors.
   * These will typically go from light to darker, in both a dark and light theme.
   */

  --jp-ui-inverse-font-color0: rgba(255, 255, 255, 1);
  --jp-ui-inverse-font-color1: rgba(255, 255, 255, 1);
  --jp-ui-inverse-font-color2: rgba(255, 255, 255, 0.7);
  --jp-ui-inverse-font-color3: rgba(255, 255, 255, 0.5);

  /* Content Fonts
   *
   * Content font variables are used for typography of user generated content.
   *
   * The font sizing here is done assuming that the body font size of --jp-content-font-size1
   * is applied to a parent element. When children elements, such as headings, are sized
   * in em all things will be computed relative to that body size.
   */

  --jp-content-line-height: 1.6;
  --jp-content-font-scale-factor: 1.2;
  --jp-content-font-size0: 0.83333em;
  --jp-content-font-size1: 14px; /* Base font size */
  --jp-content-font-size2: 1.2em;
  --jp-content-font-size3: 1.44em;
  --jp-content-font-size4: 1.728em;
  --jp-content-font-size5: 2.0736em;

  /* This gives a magnification of about 125% in presentation mode over normal. */
  --jp-content-presentation-font-size1: 17px;
  --jp-content-heading-line-height: 1;
  --jp-content-heading-margin-top: 1.2em;
  --jp-content-heading-margin-bottom: 0.8em;
  --jp-content-heading-font-weight: 500;

  /* Defaults use Material Design specification */
  --jp-content-font-color0: rgba(0, 0, 0, 1);
  --jp-content-font-color1: rgba(0, 0, 0, 0.87);
  --jp-content-font-color2: rgba(0, 0, 0, 0.54);
  --jp-content-font-color3: rgba(0, 0, 0, 0.38);
  --jp-content-link-color: var(--md-blue-900);
  --jp-content-font-family: system-ui, -apple-system, blinkmacsystemfont,
    'Segoe UI', helvetica, arial, sans-serif, 'Apple Color Emoji',
    'Segoe UI Emoji', 'Segoe UI Symbol';

  /*
   * Code Fonts
   *
   * Code font variables are used for typography of code and other monospaces content.
   */

  --jp-code-font-size: 13px;
  --jp-code-line-height: 1.3077; /* 17px for 13px base */
  --jp-code-padding: 5px; /* 5px for 13px base, codemirror highlighting needs integer px value */
  --jp-code-font-family-default: menlo, consolas, 'DejaVu Sans Mono', monospace;
  --jp-code-font-family: var(--jp-code-font-family-default);

  /* This gives a magnification of about 125% in presentation mode over normal. */
  --jp-code-presentation-font-size: 16px;

  /* may need to tweak cursor width if you change font size */
  --jp-code-cursor-width0: 1.4px;
  --jp-code-cursor-width1: 2px;
  --jp-code-cursor-width2: 4px;

  /* Layout
   *
   * The following are the main layout colors use in JupyterLab. In a light
   * theme these would go from light to dark.
   */

  --jp-layout-color0: white;
  --jp-layout-color1: white;
  --jp-layout-color2: var(--md-grey-200);
  --jp-layout-color3: var(--md-grey-400);
  --jp-layout-color4: var(--md-grey-600);

  /* Inverse Layout
   *
   * The following are the inverse layout colors use in JupyterLab. In a light
   * theme these would go from dark to light.
   */

  --jp-inverse-layout-color0: #111;
  --jp-inverse-layout-color1: var(--md-grey-900);
  --jp-inverse-layout-color2: var(--md-grey-800);
  --jp-inverse-layout-color3: var(--md-grey-700);
  --jp-inverse-layout-color4: var(--md-grey-600);

  /* Brand/accent */

  --jp-brand-color0: var(--md-blue-900);
  --jp-brand-color1: var(--md-blue-700);
  --jp-brand-color2: var(--md-blue-300);
  --jp-brand-color3: var(--md-blue-100);
  --jp-brand-color4: var(--md-blue-50);
  --jp-accent-color0: var(--md-green-900);
  --jp-accent-color1: var(--md-green-700);
  --jp-accent-color2: var(--md-green-300);
  --jp-accent-color3: var(--md-green-100);

  /* State colors (warn, error, success, info) */

  --jp-warn-color0: var(--md-orange-900);
  --jp-warn-color1: var(--md-orange-700);
  --jp-warn-color2: var(--md-orange-300);
  --jp-warn-color3: var(--md-orange-100);
  --jp-error-color0: var(--md-red-900);
  --jp-error-color1: var(--md-red-700);
  --jp-error-color2: var(--md-red-300);
  --jp-error-color3: var(--md-red-100);
  --jp-success-color0: var(--md-green-900);
  --jp-success-color1: var(--md-green-700);
  --jp-success-color2: var(--md-green-300);
  --jp-success-color3: var(--md-green-100);
  --jp-info-color0: var(--md-cyan-900);
  --jp-info-color1: var(--md-cyan-700);
  --jp-info-color2: var(--md-cyan-300);
  --jp-info-color3: var(--md-cyan-100);

  /* Cell specific styles */

  --jp-cell-padding: 5px;
  --jp-cell-collapser-width: 8px;
  --jp-cell-collapser-min-height: 20px;
  --jp-cell-collapser-not-active-hover-opacity: 0.6;
  --jp-cell-editor-background: var(--md-grey-100);
  --jp-cell-editor-border-color: var(--md-grey-300);
  --jp-cell-editor-box-shadow: inset 0 0 2px var(--md-blue-300);
  --jp-cell-editor-active-background: var(--jp-layout-color0);
  --jp-cell-editor-active-border-color: var(--jp-brand-color1);
  --jp-cell-prompt-width: 64px;
  --jp-cell-prompt-font-family: var(--jp-code-font-family-default);
  --jp-cell-prompt-letter-spacing: 0;
  --jp-cell-prompt-opacity: 1;
  --jp-cell-prompt-not-active-opacity: 0.5;
  --jp-cell-prompt-not-active-font-color: var(--md-grey-700);

  /* A custom blend of MD grey and blue 600
   * See https://meyerweb.com/eric/tools/color-blend/#546E7A:1E88E5:5:hex */
  --jp-cell-inprompt-font-color: #307fc1;

  /* A custom blend of MD grey and orange 600
   * https://meyerweb.com/eric/tools/color-blend/#546E7A:F4511E:5:hex */
  --jp-cell-outprompt-font-color: #bf5b3d;

  /* Notebook specific styles */

  --jp-notebook-padding: 10px;
  --jp-notebook-select-background: var(--jp-layout-color1);
  --jp-notebook-multiselected-color: var(--md-blue-50);

  /* The scroll padding is calculated to fill enough space at the bottom of the
  notebook to show one single-line cell (with appropriate padding) at the top
  when the notebook is scrolled all the way to the bottom. We also subtract one
  pixel so that no scrollbar appears if we have just one single-line cell in the
  notebook. This padding is to enable a 'scroll past end' feature in a notebook.
  */
  --jp-notebook-scroll-padding: calc(
    100% - var(--jp-code-font-size) * var(--jp-code-line-height) -
      var(--jp-code-padding) - var(--jp-cell-padding) - 1px
  );

  /* Rendermime styles */

  --jp-rendermime-error-background: #fdd;
  --jp-rendermime-table-row-background: var(--md-grey-100);
  --jp-rendermime-table-row-hover-background: var(--md-light-blue-50);

  /* Dialog specific styles */

  --jp-dialog-background: rgba(0, 0, 0, 0.25);

  /* Console specific styles */

  --jp-console-padding: 10px;

  /* Toolbar specific styles */

  --jp-toolbar-border-color: var(--jp-border-color1);
  --jp-toolbar-micro-height: 8px;
  --jp-toolbar-background: var(--jp-layout-color1);
  --jp-toolbar-box-shadow: 0 0 2px 0 rgba(0, 0, 0, 0.24);
  --jp-toolbar-header-margin: 4px 4px 0 4px;
  --jp-toolbar-active-background: var(--md-grey-300);

  /* Statusbar specific styles */

  --jp-statusbar-height: 24px;

  /* Input field styles */

  --jp-input-box-shadow: inset 0 0 2px var(--md-blue-300);
  --jp-input-active-background: var(--jp-layout-color1);
  --jp-input-hover-background: var(--jp-layout-color1);
  --jp-input-background: var(--md-grey-100);
  --jp-input-border-color: var(--jp-inverse-border-color);
  --jp-input-active-border-color: var(--jp-brand-color1);
  --jp-input-active-box-shadow-color: rgba(19, 124, 189, 0.3);

  /* General editor styles */

  --jp-editor-selected-background: #d9d9d9;
  --jp-editor-selected-focused-background: #d7d4f0;
  --jp-editor-cursor-color: var(--jp-ui-font-color0);

  /* Code mirror specific styles */

  --jp-mirror-editor-keyword-color: #008000;
  --jp-mirror-editor-atom-color: #88f;
  --jp-mirror-editor-number-color: #080;
  --jp-mirror-editor-def-color: #00f;
  --jp-mirror-editor-variable-color: var(--md-grey-900);
  --jp-mirror-editor-variable-2-color: rgb(0, 54, 109);
  --jp-mirror-editor-variable-3-color: #085;
  --jp-mirror-editor-punctuation-color: #05a;
  --jp-mirror-editor-property-color: #05a;
  --jp-mirror-editor-operator-color: #a2f;
  --jp-mirror-editor-comment-color: #408080;
  --jp-mirror-editor-string-color: #ba2121;
  --jp-mirror-editor-string-2-color: #708;
  --jp-mirror-editor-meta-color: #a2f;
  --jp-mirror-editor-qualifier-color: #555;
  --jp-mirror-editor-builtin-color: #008000;
  --jp-mirror-editor-bracket-color: #997;
  --jp-mirror-editor-tag-color: #170;
  --jp-mirror-editor-attribute-color: #00c;
  --jp-mirror-editor-header-color: blue;
  --jp-mirror-editor-quote-color: #090;
  --jp-mirror-editor-link-color: #00c;
  --jp-mirror-editor-error-color: #f00;
  --jp-mirror-editor-hr-color: #999;

  /*
    RTC user specific colors.
    These colors are used for the cursor, username in the editor,
    and the icon of the user.
  */

  --jp-collaborator-color1: #ffad8e;
  --jp-collaborator-color2: #dac83d;
  --jp-collaborator-color3: #72dd76;
  --jp-collaborator-color4: #00e4d0;
  --jp-collaborator-color5: #45d4ff;
  --jp-collaborator-color6: #e2b1ff;
  --jp-collaborator-color7: #ff9de6;

  /* Vega extension styles */

  --jp-vega-background: white;

  /* Sidebar-related styles */

  --jp-sidebar-min-width: 250px;

  /* Search-related styles */

  --jp-search-toggle-off-opacity: 0.5;
  --jp-search-toggle-hover-opacity: 0.8;
  --jp-search-toggle-on-opacity: 1;
  --jp-search-selected-match-background-color: rgb(245, 200, 0);
  --jp-search-selected-match-color: black;
  --jp-search-unselected-match-background-color: var(
    --jp-inverse-layout-color0
  );
  --jp-search-unselected-match-color: var(--jp-ui-inverse-font-color0);

  /* Icon colors that work well with light or dark backgrounds */
  --jp-icon-contrast-color0: var(--md-purple-600);
  --jp-icon-contrast-color1: var(--md-green-600);
  --jp-icon-contrast-color2: var(--md-pink-600);
  --jp-icon-contrast-color3: var(--md-blue-600);

  /* Button colors */
  --jp-accept-color-normal: var(--md-blue-700);
  --jp-accept-color-hover: var(--md-blue-800);
  --jp-accept-color-active: var(--md-blue-900);
  --jp-warn-color-normal: var(--md-red-700);
  --jp-warn-color-hover: var(--md-red-800);
  --jp-warn-color-active: var(--md-red-900);
  --jp-reject-color-normal: var(--md-grey-600);
  --jp-reject-color-hover: var(--md-grey-700);
  --jp-reject-color-active: var(--md-grey-800);

  /* File or activity icons and switch semantic variables */
  --jp-jupyter-icon-color: #f37626;
  --jp-notebook-icon-color: #f37626;
  --jp-json-icon-color: var(--md-orange-700);
  --jp-console-icon-background-color: var(--md-blue-700);
  --jp-console-icon-color: white;
  --jp-terminal-icon-background-color: var(--md-grey-800);
  --jp-terminal-icon-color: var(--md-grey-200);
  --jp-text-editor-icon-color: var(--md-grey-700);
  --jp-inspector-icon-color: var(--md-grey-700);
  --jp-switch-color: var(--md-grey-400);
  --jp-switch-true-position-color: var(--md-orange-900);
}
</style>
<style type="text/css">
/* Force rendering true colors when outputing to pdf */
* {
  -webkit-print-color-adjust: exact;
}

/* Misc */
a.anchor-link {
  display: none;
}

/* Input area styling */
.jp-InputArea {
  overflow: hidden;
}

.jp-InputArea-editor {
  overflow: hidden;
}

.cm-editor.cm-s-jupyter .highlight pre {
/* weird, but --jp-code-padding defined to be 5px but 4px horizontal padding is hardcoded for pre.cm-line */
  padding: var(--jp-code-padding) 4px;
  margin: 0;

  font-family: inherit;
  font-size: inherit;
  line-height: inherit;
  color: inherit;

}

.jp-OutputArea-output pre {
  line-height: inherit;
  font-family: inherit;
}

.jp-RenderedText pre {
  color: var(--jp-content-font-color1);
  font-size: var(--jp-code-font-size);
}

/* Hiding the collapser by default */
.jp-Collapser {
  display: none;
}

@page {
    margin: 0.5in; /* Margin for each printed piece of paper */
}

@media print {
  .jp-Cell-inputWrapper,
  .jp-Cell-outputWrapper {
    display: block;
  }
}
</style>
<!-- Load mathjax -->
<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/latest.js?config=TeX-AMS_CHTML-full,Safe"> </script>
<!-- MathJax configuration -->
<script type="text/x-mathjax-config">
    init_mathjax = function() {
        if (window.MathJax) {
        // MathJax loaded
            MathJax.Hub.Config({
                TeX: {
                    equationNumbers: {
                    autoNumber: "AMS",
                    useLabelIds: true
                    }
                },
                tex2jax: {
                    inlineMath: [ ['$','$'], ["\\(","\\)"] ],
                    displayMath: [ ['$$','$$'], ["\\[","\\]"] ],
                    processEscapes: true,
                    processEnvironments: true
                },
                displayAlign: 'center',
                messageStyle: 'none',
                CommonHTML: {
                    linebreaks: {
                    automatic: true
                    }
                }
            });

            MathJax.Hub.Queue(["Typeset", MathJax.Hub]);
        }
    }
    init_mathjax();
    </script>
<!-- End of mathjax configuration --><script type="module">
  document.addEventListener("DOMContentLoaded", async () => {
    const diagrams = document.querySelectorAll(".jp-Mermaid > pre.mermaid");
    // do not load mermaidjs if not needed
    if (!diagrams.length) {
      return;
    }
    const mermaid = (await import("https://cdnjs.cloudflare.com/ajax/libs/mermaid/10.7.0/mermaid.esm.min.mjs")).default;
    const parser = new DOMParser();

    mermaid.initialize({
      maxTextSize: 100000,
      maxEdges: 100000,
      startOnLoad: false,
      fontFamily: window
        .getComputedStyle(document.body)
        .getPropertyValue("--jp-ui-font-family"),
      theme: document.querySelector("body[data-jp-theme-light='true']")
        ? "default"
        : "dark",
    });

    let _nextMermaidId = 0;

    function makeMermaidImage(svg) {
      const img = document.createElement("img");
      const doc = parser.parseFromString(svg, "image/svg+xml");
      const svgEl = doc.querySelector("svg");
      const { maxWidth } = svgEl?.style || {};
      const firstTitle = doc.querySelector("title");
      const firstDesc = doc.querySelector("desc");

      img.setAttribute("src", `data:image/svg+xml,${encodeURIComponent(svg)}`);
      if (maxWidth) {
        img.width = parseInt(maxWidth);
      }
      if (firstTitle) {
        img.setAttribute("alt", firstTitle.textContent);
      }
      if (firstDesc) {
        const caption = document.createElement("figcaption");
        caption.className = "sr-only";
        caption.textContent = firstDesc.textContent;
        return [img, caption];
      }
      return [img];
    }

    async function makeMermaidError(text) {
      let errorMessage = "";
      try {
        await mermaid.parse(text);
      } catch (err) {
        errorMessage = `${err}`;
      }

      const result = document.createElement("details");
      result.className = 'jp-RenderedMermaid-Details';
      const summary = document.createElement("summary");
      summary.className = 'jp-RenderedMermaid-Summary';
      const pre = document.createElement("pre");
      const code = document.createElement("code");
      code.innerText = text;
      pre.appendChild(code);
      summary.appendChild(pre);
      result.appendChild(summary);

      const warning = document.createElement("pre");
      warning.innerText = errorMessage;
      result.appendChild(warning);
      return [result];
    }

    async function renderOneMarmaid(src) {
      const id = `jp-mermaid-${_nextMermaidId++}`;
      const parent = src.parentNode;
      let raw = src.textContent.trim();
      const el = document.createElement("div");
      el.style.visibility = "hidden";
      document.body.appendChild(el);
      let results = null;
      let output = null;
      try {
        let { svg } = await mermaid.render(id, raw, el);
        svg = cleanMermaidSvg(svg);
        results = makeMermaidImage(svg);
        output = document.createElement("figure");
        results.map(output.appendChild, output);
      } catch (err) {
        parent.classList.add("jp-mod-warning");
        results = await makeMermaidError(raw);
        output = results[0];
      } finally {
        el.remove();
      }
      parent.classList.add("jp-RenderedMermaid");
      parent.appendChild(output);
    }


    /**
     * Post-process to ensure mermaid diagrams contain only valid SVG and XHTML.
     */
    function cleanMermaidSvg(svg) {
      return svg.replace(RE_VOID_ELEMENT, replaceVoidElement);
    }


    /**
     * A regular expression for all void elements, which may include attributes and
     * a slash.
     *
     * @see https://developer.mozilla.org/en-US/docs/Glossary/Void_element
     *
     * Of these, only `<br>` is generated by Mermaid in place of `\n`,
     * but _any_ "malformed" tag will break the SVG rendering entirely.
     */
    const RE_VOID_ELEMENT =
      /<\s*(area|base|br|col|embed|hr|img|input|link|meta|param|source|track|wbr)\s*([^>]*?)\s*>/gi;

    /**
     * Ensure a void element is closed with a slash, preserving any attributes.
     */
    function replaceVoidElement(match, tag, rest) {
      rest = rest.trim();
      if (!rest.endsWith('/')) {
        rest = `${rest} /`;
      }
      return `<${tag} ${rest}>`;
    }

    void Promise.all([...diagrams].map(renderOneMarmaid));
  });
</script>
<style>
  .jp-Mermaid:not(.jp-RenderedMermaid) {
    display: none;
  }

  .jp-RenderedMermaid {
    overflow: auto;
    display: flex;
  }

  .jp-RenderedMermaid.jp-mod-warning {
    width: auto;
    padding: 0.5em;
    margin-top: 0.5em;
    border: var(--jp-border-width) solid var(--jp-warn-color2);
    border-radius: var(--jp-border-radius);
    color: var(--jp-ui-font-color1);
    font-size: var(--jp-ui-font-size1);
    white-space: pre-wrap;
    word-wrap: break-word;
  }

  .jp-RenderedMermaid figure {
    margin: 0;
    overflow: auto;
    max-width: 100%;
  }

  .jp-RenderedMermaid img {
    max-width: 100%;
  }

  .jp-RenderedMermaid-Details > pre {
    margin-top: 1em;
  }

  .jp-RenderedMermaid-Summary {
    color: var(--jp-warn-color2);
  }

  .jp-RenderedMermaid:not(.jp-mod-warning) pre {
    display: none;
  }

  .jp-RenderedMermaid-Summary > pre {
    display: inline-block;
    white-space: normal;
  }
</style>
<!-- End of mermaid configuration --></head>
<body class="jp-Notebook" data-jp-theme-light="true" data-jp-theme-name="JupyterLab Light">
<main><div class="jp-Cell jp-CodeCell jp-Notebook-cell" id="cell-id=8dcb91c9-3314-445a-94a9-1099b7814f00">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In [1]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
<div class="cm-editor cm-s-jupyter">
<div class="highlight hl-ipython3"><pre><span></span><span class="c1"># Imports</span>
<span class="kn">import</span><span class="w"> </span><span class="nn">pandas</span><span class="w"> </span><span class="k">as</span><span class="w"> </span><span class="nn">pd</span>
<span class="kn">import</span><span class="w"> </span><span class="nn">numpy</span><span class="w"> </span><span class="k">as</span><span class="w"> </span><span class="nn">np</span>
<span class="kn">import</span><span class="w"> </span><span class="nn">torch</span>
<span class="kn">import</span><span class="w"> </span><span class="nn">torch.nn</span><span class="w"> </span><span class="k">as</span><span class="w"> </span><span class="nn">nn</span>
<span class="kn">import</span><span class="w"> </span><span class="nn">torch.optim</span><span class="w"> </span><span class="k">as</span><span class="w"> </span><span class="nn">optim</span>
<span class="kn">import</span><span class="w"> </span><span class="nn">torch.nn.functional</span><span class="w"> </span><span class="k">as</span><span class="w"> </span><span class="nn">F</span>
<span class="kn">import</span><span class="w"> </span><span class="nn">matplotlib.pyplot</span><span class="w"> </span><span class="k">as</span><span class="w"> </span><span class="nn">plt</span>
<span class="kn">import</span><span class="w"> </span><span class="nn">seaborn</span><span class="w"> </span><span class="k">as</span><span class="w"> </span><span class="nn">sns</span>
<span class="kn">import</span><span class="w"> </span><span class="nn">xgboost</span><span class="w"> </span><span class="k">as</span><span class="w"> </span><span class="nn">xgb</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">torch.utils.data</span><span class="w"> </span><span class="kn">import</span> <span class="n">DataLoader</span><span class="p">,</span> <span class="n">TensorDataset</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">sklearn.preprocessing</span><span class="w"> </span><span class="kn">import</span> <span class="n">StandardScaler</span><span class="p">,</span> <span class="n">RobustScaler</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">sklearn.model_selection</span><span class="w"> </span><span class="kn">import</span> <span class="n">train_test_split</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">sklearn.metrics</span><span class="w"> </span><span class="kn">import</span> <span class="n">classification_report</span><span class="p">,</span> <span class="n">accuracy_score</span><span class="p">,</span> <span class="n">confusion_matrix</span><span class="p">,</span> <span class="n">precision_recall_curve</span><span class="p">,</span> <span class="n">roc_auc_score</span><span class="p">,</span> <span class="n">roc_curve</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">sklearn.ensemble</span><span class="w"> </span><span class="kn">import</span> <span class="n">RandomForestClassifier</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">sklearn.impute</span><span class="w"> </span><span class="kn">import</span> <span class="n">SimpleImputer</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">sklearn.utils</span><span class="w"> </span><span class="kn">import</span> <span class="n">resample</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">sklearn.preprocessing</span><span class="w"> </span><span class="kn">import</span> <span class="n">LabelEncoder</span>

<span class="c1"># GPU</span>
<span class="n">device</span> <span class="o">=</span> <span class="n">torch</span><span class="o">.</span><span class="n">device</span><span class="p">(</span><span class="s2">"cuda"</span> <span class="k">if</span> <span class="n">torch</span><span class="o">.</span><span class="n">cuda</span><span class="o">.</span><span class="n">is_available</span><span class="p">()</span> <span class="k">else</span> <span class="s2">"cpu"</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="s2">"Using device:"</span><span class="p">,</span> <span class="n">device</span><span class="p">)</span>

<span class="c1"># Constants</span>
<span class="n">lr</span> <span class="o">=</span> <span class="mf">1e-3</span>
<span class="n">batch_size</span> <span class="o">=</span> <span class="mi">64</span>
<span class="n">num_epochs</span> <span class="o">=</span> <span class="mi">75</span>

<span class="c1"># pd </span>
<span class="n">pd</span><span class="o">.</span><span class="n">set_option</span><span class="p">(</span><span class="s2">"display.max_rows"</span><span class="p">,</span> <span class="kc">None</span><span class="p">)</span>
<span class="n">pd</span><span class="o">.</span><span class="n">set_option</span><span class="p">(</span><span class="s2">"display.max_columns"</span><span class="p">,</span> <span class="kc">None</span><span class="p">)</span>
</pre></div>
</div>
</div>
</div>
</div>
<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>
<div class="jp-OutputArea jp-Cell-outputArea">
<div class="jp-OutputArea-child">
<div class="jp-OutputPrompt jp-OutputArea-prompt"></div>
<div class="jp-RenderedText jp-OutputArea-output" data-mime-type="text/plain" tabindex="0">
<pre>Using device: cuda
</pre>
</div>
</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs" id="cell-id=cd243ede-64b7-41e9-babd-8769241af6bc">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In [2]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
<div class="cm-editor cm-s-jupyter">
<div class="highlight hl-ipython3"><pre><span></span><span class="k">def</span><span class="w"> </span><span class="nf">load_datasets</span><span class="p">(</span><span class="n">base_path</span><span class="o">=</span><span class="s2">"./"</span><span class="p">):</span>
    
    <span class="n">files</span> <span class="o">=</span> <span class="p">{</span>
        <span class="s2">"train"</span><span class="p">:</span> <span class="s2">"loan_data.csv"</span><span class="p">,</span>
    <span class="p">}</span>

    <span class="n">dfs</span> <span class="o">=</span> <span class="p">{}</span>
    <span class="k">for</span> <span class="n">key</span><span class="p">,</span> <span class="n">filename</span> <span class="ow">in</span> <span class="n">files</span><span class="o">.</span><span class="n">items</span><span class="p">():</span>
        <span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s2">"Loading </span><span class="si">{</span><span class="n">filename</span><span class="si">}</span><span class="s2">..."</span><span class="p">)</span>
        <span class="n">dfs</span><span class="p">[</span><span class="n">key</span><span class="p">]</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">read_csv</span><span class="p">(</span><span class="n">base_path</span> <span class="o">+</span> <span class="n">filename</span><span class="p">)</span>
        
    <span class="k">return</span> <span class="n">dfs</span>

<span class="k">def</span><span class="w"> </span><span class="nf">dataset_summary</span><span class="p">(</span><span class="n">df</span><span class="p">,</span> <span class="n">show_counts</span><span class="o">=</span><span class="kc">True</span><span class="p">):</span>
    
    <span class="n">total_rows</span> <span class="o">=</span> <span class="nb">len</span><span class="p">(</span><span class="n">df</span><span class="p">)</span>
    <span class="n">total_duplicates</span> <span class="o">=</span> <span class="n">df</span><span class="o">.</span><span class="n">duplicated</span><span class="p">()</span><span class="o">.</span><span class="n">sum</span><span class="p">()</span>
    
    <span class="n">summary</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">DataFrame</span><span class="p">({</span>
        <span class="s2">"dtype"</span><span class="p">:</span> <span class="n">df</span><span class="o">.</span><span class="n">dtypes</span><span class="p">,</span>
        <span class="s2">"non_null_count"</span><span class="p">:</span> <span class="n">df</span><span class="o">.</span><span class="n">notna</span><span class="p">()</span><span class="o">.</span><span class="n">sum</span><span class="p">(),</span>
        <span class="s2">"missing_count"</span><span class="p">:</span> <span class="n">df</span><span class="o">.</span><span class="n">isna</span><span class="p">()</span><span class="o">.</span><span class="n">sum</span><span class="p">(),</span>
        <span class="s2">"missing_%"</span><span class="p">:</span> <span class="p">(</span><span class="n">df</span><span class="o">.</span><span class="n">isna</span><span class="p">()</span><span class="o">.</span><span class="n">mean</span><span class="p">()</span> <span class="o">*</span> <span class="mi">100</span><span class="p">)</span><span class="o">.</span><span class="n">round</span><span class="p">(</span><span class="mi">2</span><span class="p">),</span>
        <span class="s2">"unique_count"</span><span class="p">:</span> <span class="n">df</span><span class="o">.</span><span class="n">nunique</span><span class="p">(),</span>
        <span class="s2">"duplicates_in_dataset"</span><span class="p">:</span> <span class="n">total_duplicates</span> 
    <span class="p">})</span>
    
    <span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s2">"Dataset shape: </span><span class="si">{</span><span class="n">df</span><span class="o">.</span><span class="n">shape</span><span class="si">}</span><span class="s2">"</span><span class="p">)</span>
    <span class="k">if</span> <span class="n">show_counts</span><span class="p">:</span>
        <span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s2">"Total rows: </span><span class="si">{</span><span class="n">total_rows</span><span class="si">}</span><span class="s2">"</span><span class="p">)</span>
        <span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s2">"Total duplicate rows: </span><span class="si">{</span><span class="n">total_duplicates</span><span class="si">}</span><span class="s2">"</span><span class="p">)</span>
    
    <span class="n">summary</span> <span class="o">=</span> <span class="n">summary</span><span class="o">.</span><span class="n">sort_values</span><span class="p">(</span><span class="n">by</span><span class="o">=</span><span class="s2">"missing_%"</span><span class="p">,</span> <span class="n">ascending</span><span class="o">=</span><span class="kc">False</span><span class="p">)</span>
    
    <span class="k">return</span> <span class="n">summary</span>

<span class="k">def</span><span class="w"> </span><span class="nf">check_and_drop_duplicates</span><span class="p">(</span><span class="n">df</span><span class="p">,</span> <span class="n">target</span><span class="o">=</span><span class="kc">None</span><span class="p">):</span>

    <span class="n">total_duplicates</span> <span class="o">=</span> <span class="n">df</span><span class="o">.</span><span class="n">duplicated</span><span class="p">()</span><span class="o">.</span><span class="n">sum</span><span class="p">()</span>
    
    <span class="k">if</span> <span class="n">total_duplicates</span> <span class="o">&gt;</span> <span class="mi">0</span><span class="p">:</span>
        <span class="n">df_cleaned</span> <span class="o">=</span> <span class="n">df</span><span class="o">.</span><span class="n">drop_duplicates</span><span class="p">()</span>
        <span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s2">"Dropped </span><span class="si">{</span><span class="n">total_duplicates</span><span class="si">}</span><span class="s2"> duplicate rows. Remaining: </span><span class="si">{</span><span class="nb">len</span><span class="p">(</span><span class="n">df_cleaned</span><span class="p">)</span><span class="si">}</span><span class="s2">"</span><span class="p">)</span>
        
        <span class="k">if</span> <span class="n">target</span> <span class="ow">is</span> <span class="ow">not</span> <span class="kc">None</span><span class="p">:</span>
            <span class="n">target_cleaned</span> <span class="o">=</span> <span class="n">target</span><span class="o">.</span><span class="n">loc</span><span class="p">[</span><span class="n">df_cleaned</span><span class="o">.</span><span class="n">index</span><span class="p">]</span>
            <span class="k">return</span> <span class="n">df_cleaned</span><span class="p">,</span> <span class="n">target_cleaned</span>
        
        <span class="k">return</span> <span class="n">df_cleaned</span>
    <span class="k">else</span><span class="p">:</span>
        <span class="nb">print</span><span class="p">(</span><span class="s2">"No duplicate rows found."</span><span class="p">)</span>
        
        <span class="k">if</span> <span class="n">target</span> <span class="ow">is</span> <span class="ow">not</span> <span class="kc">None</span><span class="p">:</span>
            <span class="k">return</span> <span class="n">df</span><span class="p">,</span> <span class="n">target</span>
            
        <span class="k">return</span> <span class="n">df</span>

<span class="k">def</span><span class="w"> </span><span class="nf">drop_target_and_ids</span><span class="p">(</span><span class="n">df</span><span class="p">):</span>
    
    <span class="n">df_copy</span> <span class="o">=</span> <span class="n">df</span><span class="o">.</span><span class="n">copy</span><span class="p">()</span>

    <span class="n">feature_cols_to_drop</span> <span class="o">=</span><span class="p">[</span><span class="s2">"Loan_ID"</span><span class="p">,</span> <span class="s2">"Loan_Status"</span><span class="p">];</span>
    
    <span class="n">target</span> <span class="o">=</span> <span class="n">df_copy</span><span class="p">[</span><span class="s2">"Loan_Status"</span><span class="p">]</span>
    <span class="n">df_raw_features</span> <span class="o">=</span> <span class="n">df_copy</span><span class="o">.</span><span class="n">drop</span><span class="p">(</span><span class="n">columns</span><span class="o">=</span><span class="n">feature_cols_to_drop</span><span class="p">)</span>

    <span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s2">"Returning raw features and target"</span><span class="p">)</span>
    
    <span class="k">return</span> <span class="n">df_raw_features</span><span class="p">,</span> <span class="n">target</span><span class="p">,</span> <span class="n">feature_cols_to_drop</span>

<span class="k">def</span><span class="w"> </span><span class="nf">engineer_features</span><span class="p">(</span><span class="n">df</span><span class="p">):</span>
    
    <span class="n">df_engi</span> <span class="o">=</span> <span class="n">df</span><span class="o">.</span><span class="n">copy</span><span class="p">()</span>

    <span class="n">outliers_idx</span> <span class="o">=</span> <span class="n">df_engi</span><span class="p">[</span><span class="s1">'CoapplicantIncome'</span><span class="p">]</span><span class="o">.</span><span class="n">nlargest</span><span class="p">(</span><span class="mi">1</span><span class="p">)</span><span class="o">.</span><span class="n">index</span>

    <span class="n">df_train_cleaned</span> <span class="o">=</span> <span class="n">df_engi</span><span class="o">.</span><span class="n">drop</span><span class="p">(</span><span class="n">index</span><span class="o">=</span><span class="n">outliers_idx</span><span class="p">)</span>

    <span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s2">"Engineer features"</span><span class="p">)</span>
    
    <span class="k">return</span> <span class="n">df_engi</span>
    
<span class="k">def</span><span class="w"> </span><span class="nf">drop_high_missing_cols</span><span class="p">(</span><span class="n">df</span><span class="p">,</span> <span class="n">threshold</span><span class="o">=</span><span class="mf">0.3</span><span class="p">):</span>
    
    <span class="n">missing_frac</span> <span class="o">=</span> <span class="n">df</span><span class="o">.</span><span class="n">isna</span><span class="p">()</span><span class="o">.</span><span class="n">mean</span><span class="p">()</span>

    <span class="n">hm_cols_to_drop</span> <span class="o">=</span> <span class="n">missing_frac</span><span class="p">[</span><span class="n">missing_frac</span> <span class="o">&gt;</span> <span class="n">threshold</span><span class="p">]</span><span class="o">.</span><span class="n">index</span><span class="o">.</span><span class="n">tolist</span><span class="p">()</span>
    <span class="n">df_drop</span> <span class="o">=</span> <span class="n">df</span><span class="o">.</span><span class="n">drop</span><span class="p">(</span><span class="n">columns</span><span class="o">=</span><span class="n">hm_cols_to_drop</span><span class="p">)</span>

    <span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s2">"Dropping </span><span class="si">{</span><span class="nb">len</span><span class="p">(</span><span class="n">hm_cols_to_drop</span><span class="p">)</span><span class="si">}</span><span class="s2"> columns at missing threshold &gt;</span><span class="si">{</span><span class="n">threshold</span><span class="o">*</span><span class="mi">100</span><span class="si">:</span><span class="s2">.0f</span><span class="si">}</span><span class="s2">%"</span><span class="p">)</span>

    <span class="k">return</span> <span class="n">df_drop</span><span class="p">,</span> <span class="n">hm_cols_to_drop</span>

<span class="k">def</span><span class="w"> </span><span class="nf">drop_high_card_cols</span><span class="p">(</span><span class="n">df</span><span class="p">,</span> <span class="n">threshold</span><span class="o">=</span><span class="mi">50</span><span class="p">):</span>
    
    <span class="n">cat_cols</span> <span class="o">=</span> <span class="n">df</span><span class="o">.</span><span class="n">select_dtypes</span><span class="p">(</span><span class="n">include</span><span class="o">=</span><span class="p">[</span><span class="s1">'object'</span><span class="p">,</span> <span class="s1">'category'</span><span class="p">])</span><span class="o">.</span><span class="n">columns</span><span class="o">.</span><span class="n">tolist</span><span class="p">()</span>
    
    <span class="n">hc_cols_to_drop</span> <span class="o">=</span> <span class="p">[</span><span class="n">col</span> <span class="k">for</span> <span class="n">col</span> <span class="ow">in</span> <span class="n">cat_cols</span> <span class="k">if</span> <span class="n">df</span><span class="p">[</span><span class="n">col</span><span class="p">]</span><span class="o">.</span><span class="n">nunique</span><span class="p">()</span> <span class="o">&gt;</span> <span class="n">threshold</span><span class="p">]</span>
    <span class="n">df_high</span> <span class="o">=</span> <span class="n">df</span><span class="o">.</span><span class="n">drop</span><span class="p">(</span><span class="n">columns</span><span class="o">=</span><span class="n">hc_cols_to_drop</span><span class="p">,</span> <span class="n">errors</span><span class="o">=</span><span class="s1">'ignore'</span><span class="p">)</span>
    
    <span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s2">"Dropping </span><span class="si">{</span><span class="nb">len</span><span class="p">(</span><span class="n">hc_cols_to_drop</span><span class="p">)</span><span class="si">}</span><span class="s2"> high-cardinality columns (&gt; </span><span class="si">{</span><span class="n">threshold</span><span class="si">}</span><span class="s2"> unique values)"</span><span class="p">)</span>
    
    <span class="k">return</span> <span class="n">df_high</span><span class="p">,</span> <span class="n">hc_cols_to_drop</span>

<span class="k">def</span><span class="w"> </span><span class="nf">drop_correlated</span><span class="p">(</span><span class="n">df</span><span class="p">,</span> <span class="n">threshold</span><span class="o">=</span><span class="mf">0.95</span><span class="p">,</span> <span class="n">plot</span><span class="o">=</span><span class="kc">True</span><span class="p">):</span>
    
    <span class="n">df_temp</span> <span class="o">=</span> <span class="n">df</span><span class="o">.</span><span class="n">copy</span><span class="p">()</span>
    <span class="n">cat_cols</span> <span class="o">=</span> <span class="n">df_temp</span><span class="o">.</span><span class="n">select_dtypes</span><span class="p">(</span><span class="n">include</span><span class="o">=</span><span class="p">[</span><span class="s1">'object'</span><span class="p">,</span> <span class="s1">'category'</span><span class="p">])</span><span class="o">.</span><span class="n">columns</span><span class="o">.</span><span class="n">tolist</span><span class="p">()</span>
    
    <span class="k">for</span> <span class="n">col</span> <span class="ow">in</span> <span class="n">cat_cols</span><span class="p">:</span>
        <span class="n">df_temp</span><span class="p">[</span><span class="n">col</span><span class="p">]</span> <span class="o">=</span> <span class="n">df_temp</span><span class="p">[</span><span class="n">col</span><span class="p">]</span><span class="o">.</span><span class="n">astype</span><span class="p">(</span><span class="s1">'category'</span><span class="p">)</span><span class="o">.</span><span class="n">cat</span><span class="o">.</span><span class="n">codes</span>

    <span class="n">corr_matrix</span> <span class="o">=</span> <span class="n">df_temp</span><span class="o">.</span><span class="n">corr</span><span class="p">()</span><span class="o">.</span><span class="n">abs</span><span class="p">()</span>
    <span class="n">upper</span> <span class="o">=</span> <span class="n">corr_matrix</span><span class="o">.</span><span class="n">where</span><span class="p">(</span><span class="n">np</span><span class="o">.</span><span class="n">triu</span><span class="p">(</span><span class="n">np</span><span class="o">.</span><span class="n">ones</span><span class="p">(</span><span class="n">corr_matrix</span><span class="o">.</span><span class="n">shape</span><span class="p">),</span> <span class="n">k</span><span class="o">=</span><span class="mi">1</span><span class="p">)</span><span class="o">.</span><span class="n">astype</span><span class="p">(</span><span class="nb">bool</span><span class="p">))</span>
    <span class="n">corr_cols_to_drop</span> <span class="o">=</span> <span class="p">[</span><span class="n">column</span> <span class="k">for</span> <span class="n">column</span> <span class="ow">in</span> <span class="n">upper</span><span class="o">.</span><span class="n">columns</span> <span class="k">if</span> <span class="nb">any</span><span class="p">(</span><span class="n">upper</span><span class="p">[</span><span class="n">column</span><span class="p">]</span> <span class="o">&gt;</span> <span class="n">threshold</span><span class="p">)]</span>
    
    <span class="n">df_corr</span> <span class="o">=</span> <span class="n">df</span><span class="o">.</span><span class="n">drop</span><span class="p">(</span><span class="n">columns</span><span class="o">=</span><span class="n">corr_cols_to_drop</span><span class="p">)</span>
    
    <span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s2">"Dropping </span><span class="si">{</span><span class="nb">len</span><span class="p">(</span><span class="n">corr_cols_to_drop</span><span class="p">)</span><span class="si">}</span><span class="s2"> highly correlated features"</span><span class="p">)</span>
    
    <span class="k">if</span> <span class="n">plot</span><span class="p">:</span>
        <span class="n">plt</span><span class="o">.</span><span class="n">figure</span><span class="p">(</span><span class="n">figsize</span><span class="o">=</span><span class="p">(</span><span class="mi">12</span><span class="p">,</span> <span class="mi">10</span><span class="p">))</span>
        <span class="n">sns</span><span class="o">.</span><span class="n">heatmap</span><span class="p">(</span><span class="n">corr_matrix</span><span class="p">,</span> <span class="n">cmap</span><span class="o">=</span><span class="s1">'coolwarm'</span><span class="p">,</span> <span class="n">center</span><span class="o">=</span><span class="mi">0</span><span class="p">,</span> <span class="n">annot</span><span class="o">=</span><span class="kc">False</span><span class="p">,</span> <span class="n">linewidths</span><span class="o">=</span><span class="mf">0.5</span><span class="p">)</span>
        <span class="n">plt</span><span class="o">.</span><span class="n">title</span><span class="p">(</span><span class="s2">"Feature Correlation Matrix"</span><span class="p">)</span>
        <span class="n">plt</span><span class="o">.</span><span class="n">show</span><span class="p">()</span>
    
    <span class="k">return</span> <span class="n">df_corr</span><span class="p">,</span> <span class="n">corr_cols_to_drop</span>

<span class="k">def</span><span class="w"> </span><span class="nf">collapse_rare_categories</span><span class="p">(</span><span class="n">df</span><span class="p">,</span> <span class="n">min_freq</span><span class="o">=</span><span class="mf">0.005</span><span class="p">):</span>
    
    <span class="n">df_copy</span> <span class="o">=</span> <span class="n">df</span><span class="o">.</span><span class="n">copy</span><span class="p">()</span>
    
    <span class="n">cat_cols</span> <span class="o">=</span> <span class="n">df_copy</span><span class="o">.</span><span class="n">select_dtypes</span><span class="p">(</span><span class="n">include</span><span class="o">=</span><span class="p">[</span><span class="s1">'object'</span><span class="p">,</span> <span class="s1">'category'</span><span class="p">])</span><span class="o">.</span><span class="n">columns</span><span class="o">.</span><span class="n">tolist</span><span class="p">()</span>
    <span class="n">rare_maps</span> <span class="o">=</span> <span class="p">{}</span>
    <span class="n">changed</span> <span class="o">=</span> <span class="kc">False</span>

    <span class="k">for</span> <span class="n">col</span> <span class="ow">in</span> <span class="n">cat_cols</span><span class="p">:</span>
        <span class="n">freqs</span> <span class="o">=</span> <span class="n">df_copy</span><span class="p">[</span><span class="n">col</span><span class="p">]</span><span class="o">.</span><span class="n">value_counts</span><span class="p">(</span><span class="n">normalize</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>
        <span class="n">rare_cats</span> <span class="o">=</span> <span class="n">freqs</span><span class="p">[</span><span class="n">freqs</span> <span class="o">&lt;</span> <span class="n">min_freq</span><span class="p">]</span><span class="o">.</span><span class="n">index</span>
        <span class="k">if</span> <span class="nb">len</span><span class="p">(</span><span class="n">rare_cats</span><span class="p">)</span> <span class="o">&gt;</span> <span class="mi">0</span><span class="p">:</span>
            <span class="n">df_copy</span><span class="p">[</span><span class="n">col</span><span class="p">]</span> <span class="o">=</span> <span class="n">df_copy</span><span class="p">[</span><span class="n">col</span><span class="p">]</span><span class="o">.</span><span class="n">replace</span><span class="p">(</span><span class="n">rare_cats</span><span class="p">,</span> <span class="s1">'Other'</span><span class="p">)</span>
            <span class="n">rare_maps</span><span class="p">[</span><span class="n">col</span><span class="p">]</span> <span class="o">=</span> <span class="nb">set</span><span class="p">(</span><span class="n">rare_cats</span><span class="p">)</span>
            <span class="n">changed</span> <span class="o">=</span> <span class="kc">True</span>
            <span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s2">"Collapsed </span><span class="si">{</span><span class="nb">len</span><span class="p">(</span><span class="n">rare_cats</span><span class="p">)</span><span class="si">}</span><span class="s2"> rare categories in column '</span><span class="si">{</span><span class="n">col</span><span class="si">}</span><span class="s2">'"</span><span class="p">)</span>

    <span class="k">if</span> <span class="ow">not</span> <span class="n">changed</span><span class="p">:</span>
        <span class="nb">print</span><span class="p">(</span><span class="s2">"Nothing to collapse"</span><span class="p">)</span>

    <span class="k">return</span> <span class="n">df_copy</span><span class="p">,</span> <span class="p">(</span><span class="n">rare_maps</span> <span class="k">if</span> <span class="n">changed</span> <span class="k">else</span> <span class="kc">None</span><span class="p">)</span>

<span class="k">def</span><span class="w"> </span><span class="nf">impute_and_scale</span><span class="p">(</span><span class="n">df</span><span class="p">,</span> <span class="n">threshold</span><span class="o">=</span><span class="mf">1.0</span><span class="p">):</span>
    
    <span class="n">df_copy</span> <span class="o">=</span> <span class="n">df</span><span class="o">.</span><span class="n">copy</span><span class="p">()</span>
    
    <span class="n">numeric_cols</span> <span class="o">=</span> <span class="n">df_copy</span><span class="o">.</span><span class="n">select_dtypes</span><span class="p">(</span><span class="n">include</span><span class="o">=</span><span class="p">[</span><span class="s1">'number'</span><span class="p">])</span><span class="o">.</span><span class="n">columns</span><span class="o">.</span><span class="n">tolist</span><span class="p">()</span>
    <span class="n">cat_cols</span> <span class="o">=</span> <span class="n">df_copy</span><span class="o">.</span><span class="n">select_dtypes</span><span class="p">(</span><span class="n">include</span><span class="o">=</span><span class="p">[</span><span class="s1">'object'</span><span class="p">,</span> <span class="s1">'category'</span><span class="p">])</span><span class="o">.</span><span class="n">columns</span><span class="o">.</span><span class="n">tolist</span><span class="p">()</span>

    <span class="n">num_imputer</span> <span class="o">=</span> <span class="kc">None</span>
    <span class="n">cat_imputer</span> <span class="o">=</span> <span class="kc">None</span>
    <span class="n">robust_scaler</span> <span class="o">=</span> <span class="kc">None</span>
    <span class="n">std_scaler</span> <span class="o">=</span> <span class="kc">None</span>

    <span class="k">if</span> <span class="n">numeric_cols</span><span class="p">:</span>
        <span class="n">num_imputer</span> <span class="o">=</span> <span class="n">SimpleImputer</span><span class="p">(</span><span class="n">strategy</span><span class="o">=</span><span class="s1">'median'</span><span class="p">)</span>
        <span class="n">df_copy</span><span class="p">[</span><span class="n">numeric_cols</span><span class="p">]</span> <span class="o">=</span> <span class="n">num_imputer</span><span class="o">.</span><span class="n">fit_transform</span><span class="p">(</span><span class="n">df_copy</span><span class="p">[</span><span class="n">numeric_cols</span><span class="p">])</span>

    <span class="k">if</span> <span class="n">cat_cols</span><span class="p">:</span>
        <span class="n">cat_imputer</span> <span class="o">=</span> <span class="n">SimpleImputer</span><span class="p">(</span><span class="n">strategy</span><span class="o">=</span><span class="s1">'most_frequent'</span><span class="p">)</span>
        <span class="n">df_copy</span><span class="p">[</span><span class="n">cat_cols</span><span class="p">]</span> <span class="o">=</span> <span class="n">cat_imputer</span><span class="o">.</span><span class="n">fit_transform</span><span class="p">(</span><span class="n">df_copy</span><span class="p">[</span><span class="n">cat_cols</span><span class="p">])</span>

    <span class="k">if</span> <span class="n">numeric_cols</span><span class="p">:</span>
        <span class="n">skewness</span> <span class="o">=</span> <span class="n">df_copy</span><span class="p">[</span><span class="n">numeric_cols</span><span class="p">]</span><span class="o">.</span><span class="n">skew</span><span class="p">()</span><span class="o">.</span><span class="n">sort_values</span><span class="p">(</span><span class="n">ascending</span><span class="o">=</span><span class="kc">False</span><span class="p">)</span>
        <span class="n">skewed_cols</span> <span class="o">=</span> <span class="n">skewness</span><span class="p">[</span><span class="nb">abs</span><span class="p">(</span><span class="n">skewness</span><span class="p">)</span> <span class="o">&gt;</span> <span class="n">threshold</span><span class="p">]</span><span class="o">.</span><span class="n">index</span><span class="o">.</span><span class="n">tolist</span><span class="p">()</span>

        <span class="k">if</span> <span class="n">skewed_cols</span><span class="p">:</span>
            <span class="n">robust_scaler</span> <span class="o">=</span> <span class="n">RobustScaler</span><span class="p">()</span>
            <span class="n">df_copy</span><span class="p">[</span><span class="n">skewed_cols</span><span class="p">]</span> <span class="o">=</span> <span class="n">robust_scaler</span><span class="o">.</span><span class="n">fit_transform</span><span class="p">(</span><span class="n">df_copy</span><span class="p">[</span><span class="n">skewed_cols</span><span class="p">])</span><span class="o">.</span><span class="n">astype</span><span class="p">(</span><span class="n">np</span><span class="o">.</span><span class="n">float32</span><span class="p">)</span>

        <span class="n">normal_cols</span> <span class="o">=</span> <span class="p">[</span><span class="n">c</span> <span class="k">for</span> <span class="n">c</span> <span class="ow">in</span> <span class="n">numeric_cols</span> <span class="k">if</span> <span class="n">c</span> <span class="ow">not</span> <span class="ow">in</span> <span class="n">skewed_cols</span><span class="p">]</span>
        <span class="k">if</span> <span class="n">normal_cols</span><span class="p">:</span>
            <span class="n">std_scaler</span> <span class="o">=</span> <span class="n">StandardScaler</span><span class="p">()</span>
            <span class="n">df_copy</span><span class="p">[</span><span class="n">normal_cols</span><span class="p">]</span> <span class="o">=</span> <span class="n">std_scaler</span><span class="o">.</span><span class="n">fit_transform</span><span class="p">(</span><span class="n">df_copy</span><span class="p">[</span><span class="n">normal_cols</span><span class="p">])</span><span class="o">.</span><span class="n">astype</span><span class="p">(</span><span class="n">np</span><span class="o">.</span><span class="n">float32</span><span class="p">)</span>

    <span class="n">df_processed</span> <span class="o">=</span> <span class="n">df_copy</span><span class="o">.</span><span class="n">copy</span><span class="p">()</span>

    <span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s2">"Imputed and scaled features"</span><span class="p">)</span>

    <span class="k">return</span> <span class="n">df_processed</span><span class="p">,</span> <span class="n">num_imputer</span><span class="p">,</span> <span class="n">cat_imputer</span><span class="p">,</span> <span class="n">robust_scaler</span><span class="p">,</span> <span class="n">std_scaler</span>

<span class="k">def</span><span class="w"> </span><span class="nf">select_features_rf</span><span class="p">(</span><span class="n">df</span><span class="p">,</span> <span class="n">target</span><span class="p">,</span> <span class="n">n_estimators</span><span class="o">=</span><span class="mi">500</span><span class="p">,</span> <span class="n">max_depth</span><span class="o">=</span><span class="mi">10</span><span class="p">,</span> <span class="n">random_state</span><span class="o">=</span><span class="mi">42</span><span class="p">,</span> <span class="n">threshold</span><span class="o">=</span><span class="mf">0.0</span><span class="p">,</span> <span class="n">top_n</span><span class="o">=</span><span class="mi">20</span><span class="p">):</span>
    
    <span class="n">df_temp</span> <span class="o">=</span> <span class="n">df</span><span class="o">.</span><span class="n">copy</span><span class="p">()</span>
    <span class="n">cat_cols</span> <span class="o">=</span> <span class="n">df_temp</span><span class="o">.</span><span class="n">select_dtypes</span><span class="p">(</span><span class="n">include</span><span class="o">=</span><span class="p">[</span><span class="s1">'object'</span><span class="p">,</span> <span class="s1">'category'</span><span class="p">])</span><span class="o">.</span><span class="n">columns</span><span class="o">.</span><span class="n">tolist</span><span class="p">()</span>

    <span class="k">for</span> <span class="n">col</span> <span class="ow">in</span> <span class="n">cat_cols</span><span class="p">:</span>
        <span class="n">df_temp</span><span class="p">[</span><span class="n">col</span><span class="p">]</span> <span class="o">=</span> <span class="n">df_temp</span><span class="p">[</span><span class="n">col</span><span class="p">]</span><span class="o">.</span><span class="n">astype</span><span class="p">(</span><span class="s1">'category'</span><span class="p">)</span><span class="o">.</span><span class="n">cat</span><span class="o">.</span><span class="n">codes</span>

    <span class="n">rf</span> <span class="o">=</span> <span class="n">RandomForestClassifier</span><span class="p">(</span>
        <span class="n">n_estimators</span><span class="o">=</span><span class="n">n_estimators</span><span class="p">,</span>
        <span class="n">max_depth</span><span class="o">=</span><span class="n">max_depth</span><span class="p">,</span>
        <span class="n">random_state</span><span class="o">=</span><span class="n">random_state</span><span class="p">,</span>
        <span class="n">n_jobs</span><span class="o">=-</span><span class="mi">1</span>
    <span class="p">)</span>
    <span class="n">rf</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="n">df_temp</span><span class="p">,</span> <span class="n">target</span><span class="p">)</span>
    
    <span class="n">importances</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">Series</span><span class="p">(</span><span class="n">rf</span><span class="o">.</span><span class="n">feature_importances_</span><span class="p">,</span> <span class="n">index</span><span class="o">=</span><span class="n">df_temp</span><span class="o">.</span><span class="n">columns</span><span class="p">)</span><span class="o">.</span><span class="n">sort_values</span><span class="p">(</span><span class="n">ascending</span><span class="o">=</span><span class="kc">False</span><span class="p">)</span>
    <span class="n">selected_features</span> <span class="o">=</span> <span class="n">importances</span><span class="p">[</span><span class="n">importances</span> <span class="o">&gt;</span> <span class="n">threshold</span><span class="p">]</span><span class="o">.</span><span class="n">index</span><span class="o">.</span><span class="n">tolist</span><span class="p">()</span>
    <span class="n">df_selected</span> <span class="o">=</span> <span class="n">df</span><span class="p">[</span><span class="n">selected_features</span><span class="p">]</span><span class="o">.</span><span class="n">copy</span><span class="p">()</span> 

    <span class="n">top_features</span> <span class="o">=</span> <span class="n">importances</span><span class="o">.</span><span class="n">head</span><span class="p">(</span><span class="n">top_n</span><span class="p">)</span>
    <span class="n">plt</span><span class="o">.</span><span class="n">figure</span><span class="p">(</span><span class="n">figsize</span><span class="o">=</span><span class="p">(</span><span class="mi">10</span><span class="p">,</span> <span class="mi">6</span><span class="p">))</span>
    <span class="n">plt</span><span class="o">.</span><span class="n">barh</span><span class="p">(</span><span class="n">top_features</span><span class="o">.</span><span class="n">index</span><span class="p">[::</span><span class="o">-</span><span class="mi">1</span><span class="p">],</span> <span class="n">top_features</span><span class="o">.</span><span class="n">values</span><span class="p">[::</span><span class="o">-</span><span class="mi">1</span><span class="p">],</span> <span class="n">color</span><span class="o">=</span><span class="s1">'skyblue'</span><span class="p">)</span>
    <span class="n">plt</span><span class="o">.</span><span class="n">xlabel</span><span class="p">(</span><span class="s2">"Feature Importance"</span><span class="p">)</span>
    <span class="n">plt</span><span class="o">.</span><span class="n">title</span><span class="p">(</span><span class="sa">f</span><span class="s2">"Top </span><span class="si">{</span><span class="n">top_n</span><span class="si">}</span><span class="s2"> Feature Importances (Threshold=</span><span class="si">{</span><span class="n">threshold</span><span class="si">}</span><span class="s2">)"</span><span class="p">)</span>
    <span class="n">plt</span><span class="o">.</span><span class="n">tight_layout</span><span class="p">()</span>
    <span class="n">plt</span><span class="o">.</span><span class="n">show</span><span class="p">()</span>
    
    <span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s2">"Number of features before selection: </span><span class="si">{</span><span class="n">df</span><span class="o">.</span><span class="n">shape</span><span class="p">[</span><span class="mi">1</span><span class="p">]</span><span class="si">}</span><span class="s2">"</span><span class="p">)</span>
    <span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s2">"Selected </span><span class="si">{</span><span class="nb">len</span><span class="p">(</span><span class="n">selected_features</span><span class="p">)</span><span class="si">}</span><span class="s2"> features (threshold=</span><span class="si">{</span><span class="n">threshold</span><span class="si">}</span><span class="s2">)"</span><span class="p">)</span>
    
    <span class="k">return</span> <span class="n">df_selected</span><span class="p">,</span> <span class="n">selected_features</span>

<span class="k">def</span><span class="w"> </span><span class="nf">transform_val_test</span><span class="p">(</span><span class="n">df</span><span class="p">,</span> <span class="n">cols_to_drop</span><span class="p">,</span> <span class="n">selected_features</span><span class="p">,</span> <span class="n">rare_maps</span><span class="p">,</span> <span class="n">num_imputer</span><span class="p">,</span> <span class="n">cat_imputer</span><span class="p">,</span> <span class="n">robust_scaler</span><span class="p">,</span> <span class="n">std_scaler</span><span class="p">):</span>
    
    <span class="n">df_copy</span> <span class="o">=</span> <span class="n">df</span><span class="o">.</span><span class="n">copy</span><span class="p">()</span>

    <span class="k">if</span> <span class="n">cols_to_drop</span><span class="p">:</span>
        <span class="n">df_copy</span> <span class="o">=</span> <span class="n">df_copy</span><span class="o">.</span><span class="n">drop</span><span class="p">(</span><span class="n">columns</span><span class="o">=</span><span class="n">cols_to_drop</span><span class="p">,</span> <span class="n">errors</span><span class="o">=</span><span class="s1">'ignore'</span><span class="p">)</span>

    <span class="k">if</span> <span class="n">rare_maps</span><span class="p">:</span>
        <span class="k">for</span> <span class="n">col</span><span class="p">,</span> <span class="n">rare_cats</span> <span class="ow">in</span> <span class="n">rare_maps</span><span class="o">.</span><span class="n">items</span><span class="p">():</span>
            <span class="k">if</span> <span class="n">col</span> <span class="ow">in</span> <span class="n">df_copy</span><span class="o">.</span><span class="n">columns</span><span class="p">:</span>
                <span class="n">df_copy</span><span class="p">[</span><span class="n">col</span><span class="p">]</span> <span class="o">=</span> <span class="n">df_copy</span><span class="p">[</span><span class="n">col</span><span class="p">]</span><span class="o">.</span><span class="n">apply</span><span class="p">(</span><span class="k">lambda</span> <span class="n">x</span><span class="p">:</span> <span class="n">x</span> <span class="k">if</span> <span class="n">x</span> <span class="ow">not</span> <span class="ow">in</span> <span class="n">rare_cats</span> <span class="k">else</span> <span class="s1">'Other'</span><span class="p">)</span>

    <span class="n">numeric_cols</span> <span class="o">=</span> <span class="n">df_copy</span><span class="o">.</span><span class="n">select_dtypes</span><span class="p">(</span><span class="n">include</span><span class="o">=</span><span class="p">[</span><span class="s1">'number'</span><span class="p">])</span><span class="o">.</span><span class="n">columns</span><span class="o">.</span><span class="n">tolist</span><span class="p">()</span>
    <span class="k">if</span> <span class="n">numeric_cols</span><span class="p">:</span>
        <span class="n">df_copy</span><span class="p">[</span><span class="n">numeric_cols</span><span class="p">]</span> <span class="o">=</span> <span class="n">num_imputer</span><span class="o">.</span><span class="n">transform</span><span class="p">(</span><span class="n">df_copy</span><span class="p">[</span><span class="n">numeric_cols</span><span class="p">])</span>

        <span class="k">if</span> <span class="n">robust_scaler</span><span class="p">:</span>
            <span class="n">skewed_cols</span> <span class="o">=</span> <span class="n">robust_scaler</span><span class="o">.</span><span class="n">feature_names_in_</span>
            <span class="n">df_copy</span><span class="p">[</span><span class="n">skewed_cols</span><span class="p">]</span> <span class="o">=</span> <span class="n">robust_scaler</span><span class="o">.</span><span class="n">transform</span><span class="p">(</span><span class="n">df_copy</span><span class="p">[</span><span class="n">skewed_cols</span><span class="p">])</span><span class="o">.</span><span class="n">astype</span><span class="p">(</span><span class="n">np</span><span class="o">.</span><span class="n">float32</span><span class="p">)</span>

        <span class="k">if</span> <span class="n">std_scaler</span><span class="p">:</span>
            <span class="n">normal_cols</span> <span class="o">=</span> <span class="p">[</span><span class="n">c</span> <span class="k">for</span> <span class="n">c</span> <span class="ow">in</span> <span class="n">numeric_cols</span> <span class="k">if</span> <span class="n">robust_scaler</span> <span class="ow">is</span> <span class="kc">None</span> <span class="ow">or</span> <span class="n">c</span> <span class="ow">not</span> <span class="ow">in</span> <span class="n">robust_scaler</span><span class="o">.</span><span class="n">feature_names_in_</span><span class="p">]</span>
            <span class="k">if</span> <span class="n">normal_cols</span><span class="p">:</span>
                <span class="n">df_copy</span><span class="p">[</span><span class="n">normal_cols</span><span class="p">]</span> <span class="o">=</span> <span class="n">std_scaler</span><span class="o">.</span><span class="n">transform</span><span class="p">(</span><span class="n">df_copy</span><span class="p">[</span><span class="n">normal_cols</span><span class="p">])</span><span class="o">.</span><span class="n">astype</span><span class="p">(</span><span class="n">np</span><span class="o">.</span><span class="n">float32</span><span class="p">)</span>

    <span class="n">cat_cols</span> <span class="o">=</span> <span class="n">df_copy</span><span class="o">.</span><span class="n">select_dtypes</span><span class="p">(</span><span class="n">include</span><span class="o">=</span><span class="p">[</span><span class="s1">'object'</span><span class="p">,</span> <span class="s1">'category'</span><span class="p">])</span><span class="o">.</span><span class="n">columns</span><span class="o">.</span><span class="n">tolist</span><span class="p">()</span>
    <span class="k">if</span> <span class="n">cat_cols</span> <span class="ow">and</span> <span class="n">cat_imputer</span><span class="p">:</span>
        <span class="n">df_copy</span><span class="p">[</span><span class="n">cat_cols</span><span class="p">]</span> <span class="o">=</span> <span class="n">cat_imputer</span><span class="o">.</span><span class="n">transform</span><span class="p">(</span><span class="n">df_copy</span><span class="p">[</span><span class="n">cat_cols</span><span class="p">])</span>
    
    <span class="k">if</span> <span class="n">selected_features</span><span class="p">:</span>
        <span class="n">X_transformed</span> <span class="o">=</span> <span class="n">df_copy</span><span class="o">.</span><span class="n">reindex</span><span class="p">(</span><span class="n">columns</span><span class="o">=</span><span class="n">selected_features</span><span class="p">,</span> <span class="n">fill_value</span><span class="o">=</span><span class="mi">0</span><span class="p">)</span>

        <span class="k">return</span> <span class="n">X_transformed</span>

    <span class="k">return</span> <span class="n">df_copy</span>
</pre></div>
</div>
</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell" id="cell-id=7f30cd2e-7db6-495e-b168-c692582de853">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In [3]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
<div class="cm-editor cm-s-jupyter">
<div class="highlight hl-ipython3"><pre><span></span><span class="c1"># Load datasets</span>
<span class="n">dfs</span> <span class="o">=</span> <span class="n">load_datasets</span><span class="p">()</span>
<span class="n">df_train</span> <span class="o">=</span> <span class="n">dfs</span><span class="p">[</span><span class="s2">"train"</span><span class="p">]</span>
</pre></div>
</div>
</div>
</div>
</div>
<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>
<div class="jp-OutputArea jp-Cell-outputArea">
<div class="jp-OutputArea-child">
<div class="jp-OutputPrompt jp-OutputArea-prompt"></div>
<div class="jp-RenderedText jp-OutputArea-output" data-mime-type="text/plain" tabindex="0">
<pre>Loading loan_data.csv...
</pre>
</div>
</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell" id="cell-id=32e40abd-50a9-4e61-99d0-02376a16434f">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In [4]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
<div class="cm-editor cm-s-jupyter">
<div class="highlight hl-ipython3"><pre><span></span><span class="c1">#summary</span>
<span class="nb">print</span><span class="p">(</span><span class="n">dataset_summary</span><span class="p">(</span><span class="n">df_train</span><span class="p">))</span>
<span class="nb">print</span><span class="p">(</span><span class="n">df_train</span><span class="o">.</span><span class="n">head</span><span class="p">(</span><span class="mi">5</span><span class="p">))</span>
</pre></div>
</div>
</div>
</div>
</div>
<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>
<div class="jp-OutputArea jp-Cell-outputArea">
<div class="jp-OutputArea-child">
<div class="jp-OutputPrompt jp-OutputArea-prompt"></div>
<div class="jp-RenderedText jp-OutputArea-output" data-mime-type="text/plain" tabindex="0">
<pre>Dataset shape: (381, 13)
Total rows: 381
Total duplicate rows: 0
                     dtype  non_null_count  missing_count  missing_%  \
Credit_History     float64             351             30       7.87   
Self_Employed       object             360             21       5.51   
Loan_Amount_Term   float64             370             11       2.89   
Dependents          object             373              8       2.10   
Gender              object             376              5       1.31   
Education           object             381              0       0.00   
Married             object             381              0       0.00   
Loan_ID             object             381              0       0.00   
ApplicantIncome      int64             381              0       0.00   
LoanAmount         float64             381              0       0.00   
CoapplicantIncome  float64             381              0       0.00   
Property_Area       object             381              0       0.00   
Loan_Status         object             381              0       0.00   

                   unique_count  duplicates_in_dataset  
Credit_History                2                      0  
Self_Employed                 2                      0  
Loan_Amount_Term             10                      0  
Dependents                    4                      0  
Gender                        2                      0  
Education                     2                      0  
Married                       2                      0  
Loan_ID                     381                      0  
ApplicantIncome             322                      0  
LoanAmount                  101                      0  
CoapplicantIncome           182                      0  
Property_Area                 3                      0  
Loan_Status                   2                      0  
    Loan_ID Gender Married Dependents     Education Self_Employed  \
0  LP001003   Male     Yes          1      Graduate            No   
1  LP001005   Male     Yes          0      Graduate           Yes   
2  LP001006   Male     Yes          0  Not Graduate            No   
3  LP001008   Male      No          0      Graduate            No   
4  LP001013   Male     Yes          0  Not Graduate            No   

   ApplicantIncome  CoapplicantIncome  LoanAmount  Loan_Amount_Term  \
0             4583             1508.0       128.0             360.0   
1             3000                0.0        66.0             360.0   
2             2583             2358.0       120.0             360.0   
3             6000                0.0       141.0             360.0   
4             2333             1516.0        95.0             360.0   

   Credit_History Property_Area Loan_Status  
0             1.0         Rural           N  
1             1.0         Urban           Y  
2             1.0         Urban           Y  
3             1.0         Urban           Y  
4             1.0         Urban           Y  
</pre>
</div>
</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell" id="cell-id=ad14ab09-40ef-4c53-bdca-0d9cc404e658">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In [5]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
<div class="cm-editor cm-s-jupyter">
<div class="highlight hl-ipython3"><pre><span></span><span class="c1"># Drop duplicates</span>
<span class="n">df_cleaned</span> <span class="o">=</span> <span class="n">check_and_drop_duplicates</span><span class="p">(</span><span class="n">dfs</span><span class="p">[</span><span class="s2">"train"</span><span class="p">])</span>
</pre></div>
</div>
</div>
</div>
</div>
<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>
<div class="jp-OutputArea jp-Cell-outputArea">
<div class="jp-OutputArea-child">
<div class="jp-OutputPrompt jp-OutputArea-prompt"></div>
<div class="jp-RenderedText jp-OutputArea-output" data-mime-type="text/plain" tabindex="0">
<pre>No duplicate rows found.
</pre>
</div>
</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell" id="cell-id=b21bdd49-f76b-4a6b-a0e1-0cdb1926c18d">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In [6]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
<div class="cm-editor cm-s-jupyter">
<div class="highlight hl-ipython3"><pre><span></span><span class="c1"># Select targets</span>
<span class="n">df_features</span><span class="p">,</span> <span class="n">target</span><span class="p">,</span> <span class="n">feature_cols_to_drop</span> <span class="o">=</span> <span class="n">drop_target_and_ids</span><span class="p">(</span><span class="n">df_cleaned</span><span class="p">)</span>
</pre></div>
</div>
</div>
</div>
</div>
<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>
<div class="jp-OutputArea jp-Cell-outputArea">
<div class="jp-OutputArea-child">
<div class="jp-OutputPrompt jp-OutputArea-prompt"></div>
<div class="jp-RenderedText jp-OutputArea-output" data-mime-type="text/plain" tabindex="0">
<pre>Returning raw features and target
</pre>
</div>
</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell" id="cell-id=8a824380-97ff-4a2f-8a17-9be832ecb820">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In [7]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
<div class="cm-editor cm-s-jupyter">
<div class="highlight hl-ipython3"><pre><span></span><span class="nb">print</span><span class="p">(</span><span class="n">df_features</span><span class="o">.</span><span class="n">head</span><span class="p">(</span><span class="mi">5</span><span class="p">))</span>
<span class="nb">print</span><span class="p">(</span><span class="n">target</span><span class="o">.</span><span class="n">value_counts</span><span class="p">())</span>
</pre></div>
</div>
</div>
</div>
</div>
<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>
<div class="jp-OutputArea jp-Cell-outputArea">
<div class="jp-OutputArea-child">
<div class="jp-OutputPrompt jp-OutputArea-prompt"></div>
<div class="jp-RenderedText jp-OutputArea-output" data-mime-type="text/plain" tabindex="0">
<pre>  Gender Married Dependents     Education Self_Employed  ApplicantIncome  \
0   Male     Yes          1      Graduate            No             4583   
1   Male     Yes          0      Graduate           Yes             3000   
2   Male     Yes          0  Not Graduate            No             2583   
3   Male      No          0      Graduate            No             6000   
4   Male     Yes          0  Not Graduate            No             2333   

   CoapplicantIncome  LoanAmount  Loan_Amount_Term  Credit_History  \
0             1508.0       128.0             360.0             1.0   
1                0.0        66.0             360.0             1.0   
2             2358.0       120.0             360.0             1.0   
3                0.0       141.0             360.0             1.0   
4             1516.0        95.0             360.0             1.0   

  Property_Area  
0         Rural  
1         Urban  
2         Urban  
3         Urban  
4         Urban  
Loan_Status
Y    271
N    110
Name: count, dtype: int64
</pre>
</div>
</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs" id="cell-id=6819479e-6ddc-413c-a81b-89c02af1e5b2">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In [8]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
<div class="cm-editor cm-s-jupyter">
<div class="highlight hl-ipython3"><pre><span></span><span class="c1"># Split train/test</span>
<span class="n">X_train_full</span><span class="p">,</span> <span class="n">X_test</span><span class="p">,</span> <span class="n">y_train_full</span><span class="p">,</span> <span class="n">y_test</span> <span class="o">=</span> <span class="n">train_test_split</span><span class="p">(</span>
    <span class="n">df_features</span><span class="p">,</span> <span class="n">target</span><span class="p">,</span> <span class="n">test_size</span><span class="o">=</span><span class="mf">0.2</span><span class="p">,</span> <span class="n">stratify</span><span class="o">=</span><span class="n">target</span><span class="p">,</span> <span class="n">random_state</span><span class="o">=</span><span class="mi">42</span>
<span class="p">)</span>

<span class="c1"># Split train/val</span>
<span class="n">X_train</span><span class="p">,</span> <span class="n">X_val</span><span class="p">,</span> <span class="n">y_train</span><span class="p">,</span> <span class="n">y_val</span> <span class="o">=</span> <span class="n">train_test_split</span><span class="p">(</span>
    <span class="n">X_train_full</span><span class="p">,</span> <span class="n">y_train_full</span><span class="p">,</span> <span class="n">test_size</span><span class="o">=</span><span class="mf">0.2</span><span class="p">,</span> <span class="n">stratify</span><span class="o">=</span><span class="n">y_train_full</span><span class="p">,</span> <span class="n">random_state</span><span class="o">=</span><span class="mi">42</span>
<span class="p">)</span>
</pre></div>
</div>
</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell" id="cell-id=7e62234a-5e58-442a-8717-235a0e6e2875">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In [9]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
<div class="cm-editor cm-s-jupyter">
<div class="highlight hl-ipython3"><pre><span></span><span class="n">numeric_df</span> <span class="o">=</span> <span class="n">X_train</span><span class="o">.</span><span class="n">select_dtypes</span><span class="p">(</span><span class="n">include</span><span class="o">=</span><span class="p">[</span><span class="s1">'int64'</span><span class="p">,</span> <span class="s1">'float64'</span><span class="p">])</span>

<span class="n">plt</span><span class="o">.</span><span class="n">figure</span><span class="p">(</span><span class="n">figsize</span><span class="o">=</span><span class="p">(</span><span class="mi">15</span><span class="p">,</span> <span class="mi">6</span><span class="p">))</span>
<span class="n">sns</span><span class="o">.</span><span class="n">boxplot</span><span class="p">(</span><span class="n">data</span><span class="o">=</span><span class="n">numeric_df</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">title</span><span class="p">(</span><span class="s2">"Boxplot for All Numeric Features"</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">xticks</span><span class="p">(</span><span class="n">rotation</span><span class="o">=</span><span class="mi">45</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">show</span><span class="p">()</span>
</pre></div>
</div>
</div>
</div>
</div>
<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>
<div class="jp-OutputArea jp-Cell-outputArea">
<div class="jp-OutputArea-child">
<div class="jp-OutputPrompt jp-OutputArea-prompt"></div>
<div class="jp-RenderedImage jp-OutputArea-output" tabindex="0">
<img alt="No description has been provided for this image" class="" src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAABNQAAAJrCAYAAAAoBM4TAAAAOnRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjEwLjUsIGh0dHBzOi8vbWF0cGxvdGxpYi5vcmcvWftoOwAAAAlwSFlzAAAPYQAAD2EBqD+naQAAo71JREFUeJzs3Xt8z/X///H7e7OD07ZGNsuMUU455DQrjJLlUCmlJM2pA6Mcyrk5JIrkOCMVHSwhhyKnHKaYQ+QzhygMFRvRNmY23nv9/ui313fvLLyEt9nterm8L/V+vR7v1/vxevO29/u+5+v5tBmGYQgAAAAAAADAVXFxdgMAAAAAAABAfkKgBgAAAAAAAFhAoAYAAAAAAABYQKAGAAAAAAAAWECgBgAAAAAAAFhAoAYAAAAAAABYQKAGAAAAAAAAWECgBgAAAAAAAFhAoAYAAAAAAABYQKAGAABuSTabTcOHD78pz7VixQrVqlVLnp6estlsSklJuSnPezX++TrMnj1bNptNhw8fdlpPt4LDhw/LZrNp9uzZzm4FAAAUQARqAAAUMDmBTO5bqVKl1LRpUy1fvtzZ7f1ne/fu1fDhw686cDp16pTatWunwoULKzo6Wp999pmKFi16Y5v8/6ZNmyabzaaQkJDrfuzhw4fLZrPJz89P586du2R/uXLl1Lp16+v+vPlRTjiX161BgwY35DmPHTum4cOHa+fOnTfk+AAA4MYq5OwGAACAc4wcOVLly5eXYRhKTk7W7Nmz1bJlS33zzTf5OmjZu3evRowYoSZNmqhcuXJXrN+2bZvOnDmjt956S82aNbvxDeYyZ84clStXTlu3btWBAwdUsWLF6/4cJ06cUExMjPr163fdj+1MQUFBysjIkJub23U7Zvv27dWyZUuHbXfeeed1O35ux44d04gRI1SuXDnVqlXrhjwHAAC4cQjUAAAooFq0aKG6deua97t27So/Pz998cUX+TpQs+rEiROSJB8fn+t2zPT09CuOcktMTNSmTZu0cOFCvfzyy5ozZ46GDRt23XrIUatWLY0bN049evRQ4cKFr/vxb7aLFy8qOztb7u7u8vT0vK7Hrl27tp5//vnresyb7fz583J3d5eLCxeiAABwI/GTFgAASPo7UCpcuLAKFXL8fVt6err69eunwMBAeXh4qFKlSnrvvfdkGIYkKSMjQ5UrV1blypWVkZFhPu706dMqXbq07r//ftntdklSp06dVKxYMR06dEjh4eEqWrSoAgICNHLkSPN4l/PTTz+pRYsW8vLyUrFixfTQQw9p8+bN5v7Zs2fr6aefliQ1bdrUvGxv/fr1eR6vSZMmioiIkCTVq1dPNptNnTp1MvfPnz9fderUUeHChVWyZEk9//zz+uOPPxyOkXNOBw8eVMuWLVW8eHF16NDhiucyZ84c3XHHHWrVqpWeeuopzZkz54qPuRZRUVFKTk5WTEzMZevWr1+f52uV11xlOed89OhRtW7dWsWKFdNdd92l6OhoSdKuXbv04IMPqmjRogoKClJsbOwlz5eSkqLevXubf68qVqyod999V9nZ2Zc893vvvaeJEyeqQoUK8vDw0N69e/91DrV9+/apXbt2uvPOO1W4cGFVqlRJQ4YMsfai/Yt9+/bpqaeekq+vrzw9PVW3bl19/fXXDjWnT5/W66+/rurVq6tYsWLy8vJSixYt9L///c+sWb9+verVqydJ6ty5s/n3NOdcypUr5/D3MEeTJk3UpEkTh+PYbDbNnTtXQ4cO1V133aUiRYooLS1NkrRlyxY98sgj8vb2VpEiRRQWFqaNGzc6HPPMmTPq3bu3ypUrJw8PD5UqVUoPP/ywduzYcR1eMQAAbl+MUAMAoIBKTU3Vn3/+KcMwdOLECU2ZMkVnz551GKFjGIYee+wxrVu3Tl27dlWtWrW0cuVKvfHGG/rjjz80YcIEFS5cWJ988okeeOABDRkyRO+//74kKTIyUqmpqZo9e7ZcXV3NY9rtdj3yyCNq0KCBxo4dqxUrVmjYsGG6ePGiRo4c+a/97tmzR40aNZKXl5f69+8vNzc3zZgxQ02aNFFcXJxCQkLUuHFjvfrqq5o8ebIGDx6sKlWqSJL5338aMmSIKlWqpA8++MC8BLZChQqS/g7nOnfurHr16mnMmDFKTk7WpEmTtHHjRv30008OI9ouXryo8PBwNWzYUO+9956KFClyxdd/zpw5evLJJ+Xu7q727dsrJiZG27ZtM4OW66VRo0Z68MEHNXbsWHXv3v26jVKz2+1q0aKFGjdurLFjx2rOnDnq2bOnihYtqiFDhqhDhw568sknNX36dL3wwgsKDQ1V+fLlJUnnzp1TWFiY/vjjD7388ssqW7asNm3apEGDBun48eOaOHGiw3PNmjVL58+f10svvSQPDw/5+vo6BG85EhIS1KhRI7m5uemll15SuXLldPDgQX3zzTd6++23r3hO586d059//umwzdvbW25ubtqzZ48eeOAB3XXXXRo4cKCKFi2qefPmqU2bNvrqq6/0xBNPSJIOHTqkxYsX6+mnn1b58uWVnJysGTNmKCwsTHv37lVAQICqVKmikSNHKioqSi+99JIaNWokSbr//vuv5Y9Cb731ltzd3fX6668rMzNT7u7uWrt2rVq0aKE6depo2LBhcnFx0axZs/Tggw/q+++/V/369SVJr7zyihYsWKCePXuqatWqOnXqlH744Qf9/PPPql279jX1AwBAgWAAAIACZdasWYakS24eHh7G7NmzHWoXL15sSDJGjRrlsP2pp54ybDabceDAAXPboEGDDBcXF2PDhg3G/PnzDUnGxIkTHR4XERFhSDJ69eplbsvOzjZatWpluLu7GydPnjS3SzKGDRtm3m/Tpo3h7u5uHDx40Nx27Ngxo3jx4kbjxo3NbTnPvW7dOkuvx7Zt28xtWVlZRqlSpYx7773XyMjIMLcvXbrUkGRERUVdck4DBw68quczDMP48ccfDUnG6tWrzdegTJkyxmuvvXZJ7T9fh5x+ExMTL/scw4YNMyQZJ0+eNOLi4gxJxvvvv2/uDwoKMlq1amXeX7duXZ6vW2JioiHJmDVr1iXnPHr0aHPbX3/9ZRQuXNiw2WzG3Llzze379u275Bzeeusto2jRosYvv/zi8FwDBw40XF1djaNHjzo8t5eXl3HixIkr9tW4cWOjePHixpEjRxxqs7OzL/ta5Rwrr1vO6/HQQw8Z1atXN86fP+9w3Pvvv9+4++67zW3nz5837Hb7Jcf38PAwRo4caW7btm3bJf3nCAoKMiIiIi7ZHhYWZoSFhZn3c/7MgoODjXPnzjn0dffddxvh4eEO537u3DmjfPnyxsMPP2xu8/b2NiIjIy/7+gAAgEtxyScAAAVUdHS0Vq9erdWrV+vzzz9X06ZN1a1bNy1cuNCs+fbbb+Xq6qpXX33V4bH9+vWTYRgOq4IOHz5c1apVU0REhHr06KGwsLBLHpejZ8+e5v/bbDb17NlTWVlZ+u677/Kst9vtWrVqldq0aaPg4GBze+nSpfXcc8/phx9+MC9zux5+/PFHnThxQj169HCYp6tVq1aqXLmyli1bdsljunfvftXHnzNnjvz8/NS0aVNJf78GzzzzjObOnWteHns9NW7cWE2bNtXYsWMdLsv9r7p162b+v4+PjypVqqSiRYuqXbt25vZKlSrJx8dHhw4dMrfNnz9fjRo10h133KE///zTvDVr1kx2u10bNmxweJ62bdtecXGAkydPasOGDerSpYvKli3rsM9ms13V+bz00kvmeyLnVrNmTZ0+fVpr165Vu3btdObMGbPfU6dOKTw8XL/++qt5KbCHh4c5f5ndbtepU6dUrFgxVapU6YZdRhkREeEw8nDnzp369ddf9dxzz+nUqVNmv+np6XrooYe0YcMGc4Sfj4+PtmzZomPHjt2Q3gAAuF1xyScAAAVU/fr1HRYlaN++ve677z717NlTrVu3lru7u44cOaKAgAAVL17c4bE5l1AeOXLE3Obu7q6PP/5Y9erVk6enp2bNmpVnkOHi4uIQiknSPffcI+nvObPycvLkSZ07d06VKlW6ZF+VKlWUnZ2t3377TdWqVbu6k7+CnPPK6/kqV66sH374wWFboUKFVKZMmas6tt1u19y5c9W0aVMlJiaa20NCQjR+/HitWbNGzZs3/w/d52348OEKCwvT9OnT1adPn/98PE9Pz0tCLm9vb5UpU+aSP3dvb2/99ddf5v1ff/1VCQkJ/xqS5SwUkSPnUtHLyQns7r333qvqPy933313niu9bt26VYZh6M0339Sbb76Z52NPnDihu+66S9nZ2Zo0aZKmTZumxMREh4C0RIkS19zb5fzz9fn1118lyZwfMC+pqam64447NHbsWEVERCgwMFB16tRRy5Yt9cILL1zyHgUAAI4I1AAAgKS/g66mTZtq0qRJ+vXXX68pnFq5cqWkv1ca/PXXX68qCLkd5B6VdCVr167V8ePHNXfuXM2dO/eS/XPmzLkhgVrjxo3VpEkTjR07Vq+88sol+/9tFNe/jZjLPS/e1Ww3ci06kZ2drYcfflj9+/fPszYnYM3h7NVJc0Zzvf766woPD8+zpmLFipKk0aNH680331SXLl301ltvydfXVy4uLurdu3ee877l5XJ/Fnm9vv98fXKeZ9y4capVq1aexypWrJgkqV27dmrUqJEWLVqkVatWady4cXr33Xe1cOFCtWjR4qr6BQCgICJQAwAAposXL0qSzp49K0kKCgrSd999pzNnzjiMUtu3b5+5P0dCQoJGjhypzp07a+fOnerWrZt27dolb29vh+fIzs7WoUOHHEKTX375RdLfqxvm5c4771SRIkW0f//+S/bt27dPLi4uCgwMlHT1l/ddTs557d+/Xw8++KDDvv379zuct1Vz5sxRqVKlzBUxc1u4cKEWLVqk6dOn35AQafjw4WrSpIlmzJhxyb477rhD0t+rb+aWexTi9VKhQgWdPXs2z9Fg1ypnRNXu3buv2zH/eWw3N7cr9rxgwQI1bdpUH330kcP2lJQUlSxZ0rx/ub+nd9xxxyV/DtLffxZXM3IsZ2ENLy+vq3qNS5curR49eqhHjx46ceKEateurbfffptADQCAy2AONQAAIEm6cOGCVq1aJXd3d/OSzpYtW8put2vq1KkOtRMmTJDNZjO/cF+4cEGdOnVSQECAJk2apNmzZys5OflfLy3MfTzDMDR16lS5ubnpoYceyrPe1dVVzZs315IlSxwuC01OTlZsbKwaNmwoLy8vSVLRokUlXRoMWVG3bl2VKlVK06dPV2Zmprl9+fLl+vnnn9WqVatrOm5GRoYWLlyo1q1b66mnnrrk1rNnT505c0Zff/31Nfd+OWFhYWrSpIneffddnT9/3mFfUFCQXF1dL5m/bNq0ade9j3bt2ik+Pt4c0ZhbSkqKGexaceedd6px48b6+OOPdfToUYd9uUfHXYtSpUqZQeTx48cv2X/y5Enz/11dXS95vvnz55tzrOW43N/TChUqaPPmzcrKyjK3LV26VL/99ttV9VunTh1VqFBB7733nhmO59Wv3W5Xamqqw75SpUopICDA4e89AAC4FCPUAAAooJYvX26ONDtx4oRiY2P166+/auDAgWY49eijj6pp06YaMmSIDh8+rJo1a2rVqlVasmSJevfubY6EGTVqlHbu3Kk1a9aoePHiqlGjhqKiojR06FA99dRTatmypfm8np6eWrFihSIiIhQSEqLly5dr2bJlGjx48GUnnh81apRWr16thg0bqkePHipUqJBmzJihzMxMjR071qyrVauWXF1d9e677yo1NVUeHh568MEHVapUqat+bdzc3PTuu++qc+fOCgsLU/v27ZWcnKxJkyapXLly1zwH2ddff60zZ87osccey3N/gwYNdOedd2rOnDl65plnruk5rmTYsGHmYgi5eXt76+mnn9aUKVNks9lUoUIFLV269JL5zK6HN954Q19//bVat26tTp06qU6dOkpPT9euXbu0YMECHT582GE019WaPHmyGjZsqNq1a+ull15S+fLldfjwYS1btkw7d+78Tz1HR0erYcOGql69ul588UUFBwcrOTlZ8fHx+v333/W///1PktS6dWtzpOb999+vXbt2ac6cOZeMLKtQoYJ8fHw0ffp0FS9eXEWLFlVISIjKly+vbt26acGCBXrkkUfUrl07HTx4UJ9//rn5frsSFxcXffjhh2rRooWqVaumzp0766677tIff/yhdevWycvLS998843OnDmjMmXK6KmnnlLNmjVVrFgxfffdd9q2bZvGjx//n14vAABue85cYhQAANx8s2bNMiQ53Dw9PY1atWoZMTExRnZ2tkP9mTNnjD59+hgBAQGGm5ubcffddxvjxo0z67Zv324UKlTI6NWrl8PjLl68aNSrV88ICAgw/vrrL8MwDCMiIsIoWrSocfDgQaN58+ZGkSJFDD8/P2PYsGGG3W53eLwkY9iwYQ7bduzYYYSHhxvFihUzihQpYjRt2tTYtGnTJec4c+ZMIzg42HB1dTUkGevWrbvi67Ft27ZL9n355ZfGfffdZ3h4eBi+vr5Ghw4djN9//92hJuecrsajjz5qeHp6Gunp6f9a06lTJ8PNzc34888/DcO49HXI6TcxMfGyzzVs2DBDknHy5MlL9oWFhRmSjFatWjlsP3nypNG2bVujSJEixh133GG8/PLLxu7duw1JxqxZs8y6fzvnsLAwo1q1apdsDwoKuuS5zpw5YwwaNMioWLGi4e7ubpQsWdK4//77jffee8/IysoyDMMwEhMTDUnGuHHjLjlmzr7cfRmGYezevdt44oknDB8fH8PT09OoVKmS8eabb/7r63Sl58nt4MGDxgsvvGD4+/sbbm5uxl133WW0bt3aWLBggVlz/vx5o1+/fkbp0qWNwoULGw888IARHx9vhIWFGWFhYQ7HW7JkiVG1alWjUKFCl5zL+PHjjbvuusvw8PAwHnjgAePHH3+85Bjr1q0zJBnz58/Ps9+ffvrJePLJJ40SJUoYHh4eRlBQkNGuXTtjzZo1hmEYRmZmpvHGG28YNWvWNIoXL24ULVrUqFmzpjFt2rTLvg4AAMAwbIbxH8fAAwAAXKVOnTppwYIFeV6GBgAAAOQXzKEGAAAAAAAAWECgBgAAAAAAAFhAoAYAAAAAAABYwBxqAAAAAAAAgAWMUAMAAAAAAAAsIFADAAAAAAAALCjk7AacKTs7W8eOHVPx4sVls9mc3Q4AAAAAAACcxDAMnTlzRgEBAXJxufwYtAIdqB07dkyBgYHObgMAAAAAAAC3iN9++01lypS5bE2BDtSKFy8u6e8XysvLy8ndAAAAAAAAwFnS0tIUGBho5kWXYylQi4mJUUxMjA4fPixJqlatmqKiotSiRQtJUpMmTRQXF+fwmJdfflnTp0837x89elTdu3fXunXrVKxYMUVERGjMmDEqVOj/Wlm/fr369u2rPXv2KDAwUEOHDlWnTp0cjhsdHa1x48YpKSlJNWvW1JQpU1S/fn0rp2Ne5unl5UWgBgAAAAAAgKuaFszSogRlypTRO++8o+3bt+vHH3/Ugw8+qMcff1x79uwxa1588UUdP37cvI0dO9bcZ7fb1apVK2VlZWnTpk365JNPNHv2bEVFRZk1iYmJatWqlZo2baqdO3eqd+/e6tatm1auXGnWfPnll+rbt6+GDRumHTt2qGbNmgoPD9eJEyesnA4AAAAAAABgmc0wDOO/HMDX11fjxo1T165d1aRJE9WqVUsTJ07Ms3b58uVq3bq1jh07Jj8/P0nS9OnTNWDAAJ08eVLu7u4aMGCAli1bpt27d5uPe/bZZ5WSkqIVK1ZIkkJCQlSvXj1NnTpV0t+LCwQGBqpXr14aOHDgVfeelpYmb29vpaamMkINAAAAAACgALOSE1kaoZab3W7X3LlzlZ6ertDQUHP7nDlzVLJkSd17770aNGiQzp07Z+6Lj49X9erVzTBNksLDw5WWlmaOcouPj1ezZs0cnis8PFzx8fGSpKysLG3fvt2hxsXFRc2aNTNr/k1mZqbS0tIcbgAAAAAAAIAVlhcl2LVrl0JDQ3X+/HkVK1ZMixYtUtWqVSVJzz33nIKCghQQEKCEhAQNGDBA+/fv18KFCyVJSUlJDmGaJPN+UlLSZWvS0tKUkZGhv/76S3a7Pc+affv2Xbb3MWPGaMSIEVZPGQAAAAAAADBZDtQqVaqknTt3KjU1VQsWLFBERITi4uJUtWpVvfTSS2Zd9erVVbp0aT300EM6ePCgKlSocF0bvxaDBg1S3759zfs5qzcAAAAAAAAAV8tyoObu7q6KFStKkurUqaNt27Zp0qRJmjFjxiW1ISEhkqQDBw6oQoUK8vf319atWx1qkpOTJUn+/v7mf3O25a7x8vJS4cKF5erqKldX1zxrco7xbzw8POTh4WHhbAEAAAAAAABH1zyHWo7s7GxlZmbmuW/nzp2SpNKlS0uSQkNDtWvXLofVOFevXi0vLy/zstHQ0FCtWbPG4TirV68252lzd3dXnTp1HGqys7O1Zs0ah7ncAAAAAAAAgBvB0gi1QYMGqUWLFipbtqzOnDmj2NhYrV+/XitXrtTBgwcVGxurli1bqkSJEkpISFCfPn3UuHFj1ahRQ5LUvHlzVa1aVR07dtTYsWOVlJSkoUOHKjIy0hw59sorr2jq1Knq37+/unTporVr12revHlatmyZ2Uffvn0VERGhunXrqn79+po4caLS09PVuXPn6/jSAAAAAAAAAJeyFKidOHFCL7zwgo4fPy5vb2/VqFFDK1eu1MMPP6zffvtN3333nRluBQYGqm3btho6dKj5eFdXVy1dulTdu3dXaGioihYtqoiICI0cOdKsKV++vJYtW6Y+ffpo0qRJKlOmjD788EOFh4ebNc8884xOnjypqKgoJSUlqVatWlqxYsUlCxUAAAAAAAAA15vNMAzD2U04S1pamry9vZWamiovLy9ntwMAAAAAAAAnsZIT/ec51AAAAAAAAICChEANAAAAAAAAsIBADQAAAAAAALDA0qIEAABYZbfblZCQoFOnTqlEiRKqUaOGXF1dnd0WAAAAAFwzAjUAwA0TFxen6OhoJSUlmdv8/f0VGRmpsLAwJ3YGAAAAANeOSz4BADdEXFycoqKiFBwcrJiYGK1YsUIxMTEKDg5WVFSU4uLinN0iAAAAAFwTm2EYhrObcBYry6ECAK6e3W5X+/btFRwcrNGjR8vF5f9+f5Odna3BgwcrMTFRsbGxXP4JAAAA4JZgJSdihBoA4LpLSEhQUlKSOnbs6BCmSZKLi4uef/55HT9+XAkJCU7qEAAAAACuHYEaAOC6O3XqlCSpfPnyee4PDg52qAMAAACA/IRADQBw3ZUoUUKSlJiYmOf+Q4cOOdQBAAAAQH5CoAYAuO5q1Kghf39/ffbZZ8rOznbYl52drc8//1ylS5dWjRo1nNQhAAAAAFw7AjUAwHXn6uqqyMhIxcfHa/Dgwdq9e7fOnTun3bt3a/DgwYqPj1ePHj1YkAAAAABAvsQqn6zyCQA3TFxcnKKjo5WUlGRuK126tHr06KGwsDAndgYAAAAAjqzkRARqBGoAcEPZ7XYlJCTo1KlTKlGihGrUqMHINAAAAAC3HCs5UaGb1BMAoIBydXXVfffd5+w2AAAAAOC6YQ41AAAAAAAAwAICNQAAAAAAAMACAjUAAAAAAADAAgI1AAAAAAAAwAICNQAAAAAAAMACAjUAAAAAAADAAgI1AAAAAAAAwAICNQAAAAAAAMACAjUAAAAAAADAAgI1AAAAAAAAwAICNQAAAAAAAMACAjUAAAAAAADAAgI1AAAAAAAAwAICNQAAAAAAAMACAjUAAAAAAADAAgI1AAAAAAAAwAICNQAAAAAAAMACAjUAAAAAAADAAgI1AAAAAAAAwAICNQAAAAAAAMACAjUAAAAAAADAAgI1AAAAAAAAwAICNQAAAAAAAMACAjUAAAAAAADAAgI1AAAAAAAAwAICNQAAAAAAAMACAjUAAAAAAADAAgI1AAAAAAAAwAICNQAAAAAAAMACAjUAAAAAAADAAgI1AAAAAAAAwAICNQAAAAAAAMACAjUAAAAAAADAAgI1AAAAAAAAwAICNQAAAAAAAMACAjUAAAAAAADAAkuBWkxMjGrUqCEvLy95eXkpNDRUy5cvN/efP39ekZGRKlGihIoVK6a2bdsqOTnZ4RhHjx5Vq1atVKRIEZUqVUpvvPGGLl686FCzfv161a5dWx4eHqpYsaJmz559SS/R0dEqV66cPD09FRISoq1bt1o5FQAAAAAAAOCaWArUypQpo3feeUfbt2/Xjz/+qAcffFCPP/649uzZI0nq06ePvvnmG82fP19xcXE6duyYnnzySfPxdrtdrVq1UlZWljZt2qRPPvlEs2fPVlRUlFmTmJioVq1aqWnTptq5c6d69+6tbt26aeXKlWbNl19+qb59+2rYsGHasWOHatasqfDwcJ04ceK/vh4AAAAAAADAZdkMwzD+ywF8fX01btw4PfXUU7rzzjsVGxurp556SpK0b98+ValSRfHx8WrQoIGWL1+u1q1b69ixY/Lz85MkTZ8+XQMGDNDJkyfl7u6uAQMGaNmyZdq9e7f5HM8++6xSUlK0YsUKSVJISIjq1aunqVOnSpKys7MVGBioXr16aeDAgVfde1pamry9vZWamiovL6//8jIAAAAAAAAgH7OSE13zHGp2u11z585Venq6QkNDtX37dl24cEHNmjUzaypXrqyyZcsqPj5ekhQfH6/q1aubYZokhYeHKy0tzRzlFh8f73CMnJqcY2RlZWn79u0ONS4uLmrWrJlZ828yMzOVlpbmcAMAAAAAAACssByo7dq1S8WKFZOHh4deeeUVLVq0SFWrVlVSUpLc3d3l4+PjUO/n56ekpCRJUlJSkkOYlrM/Z9/latLS0pSRkaE///xTdrs9z5qcY/ybMWPGyNvb27wFBgZaPX0AAAAAAAAUcJYDtUqVKmnnzp3asmWLunfvroiICO3du/dG9HbdDRo0SKmpqebtt99+c3ZLAAAAAAAAyGcKWX2Au7u7KlasKEmqU6eOtm3bpkmTJumZZ55RVlaWUlJSHEapJScny9/fX5Lk7+9/yWqcOauA5q7558qgycnJ8vLyUuHCheXq6ipXV9c8a3KO8W88PDzk4eFh9ZQBAAAAAAAA0zXPoZYjOztbmZmZqlOnjtzc3LRmzRpz3/79+3X06FGFhoZKkkJDQ7Vr1y6H1ThXr14tLy8vVa1a1azJfYycmpxjuLu7q06dOg412dnZWrNmjVkDAAAAAAAA3CiWRqgNGjRILVq0UNmyZXXmzBnFxsZq/fr1Wrlypby9vdW1a1f17dtXvr6+8vLyUq9evRQaGqoGDRpIkpo3b66qVauqY8eOGjt2rJKSkjR06FBFRkaaI8deeeUVTZ06Vf3791eXLl20du1azZs3T8uWLTP76Nu3ryIiIlS3bl3Vr19fEydOVHp6ujp37nwdXxoAAAAAAADgUpYCtRMnTuiFF17Q8ePH5e3trRo1amjlypV6+OGHJUkTJkyQi4uL2rZtq8zMTIWHh2vatGnm411dXbV06VJ1795doaGhKlq0qCIiIjRy5Eizpnz58lq2bJn69OmjSZMmqUyZMvrwww8VHh5u1jzzzDM6efKkoqKilJSUpFq1amnFihWXLFQAAAAAAAAAXG82wzAMZzfhLGlpafL29lZqaqq8vLyc3Q4AAAAAAACcxEpO9J/nUAMAAAAAAAAKEgI1AAAAAAAAwAICNQAAAAAAAMACAjUAAAAAAADAAgI1AAAAAAAAwAICNQAAAAAAAMACAjUAAAAAAADAAgI1AAAAAAAAwAICNQAAAAAAAMACAjUAAAAAAADAAgI1AAAAAAAAwAICNQAAAAAAAMACAjUAAAAAAADAAgI1AAAAAAAAwAICNQAAAAAAAMACAjUAAAAAAADAAgI1AAAAAAAAwAICNQAAAAAAAMACAjUAAAAAAADAAgI1AAAAAAAAwAICNQAAAAAAAMACAjUAAAAAAADAAgI1AAAAAAAAwAICNQAAAAAAAMACAjUAAAAAAADAAgI1AAAAAAAAwAICNQAAAAAAAMACAjUAAAAAAADAAgI1AAAAAAAAwAICNQAAAAAAAMACAjUAAAAAAADAAgI1AAAAAAAAwAICNQAAAAAAAMACAjUAAAAAAADAAgI1AAAAAAAAwAICNQAAAAAAAMACAjUAAAAAAADAAgI1AAAAAAAAwAICNQAAAAAAAMACAjUAAAAAAADAAgI1AAAAAAAAwAICNQAAAAAAAMACAjUAAAAAAADAAgI1AAAAAAAAwAICNQAAAAAAAMACAjUAAAAAAADAAgI1AAAAAAAAwAICNQAAAAAAAMACAjUAAAAAAADAAgI1AAAAAAAAwAICNQAAAAAAAMACAjUAAAAAAADAAkuB2pgxY1SvXj0VL15cpUqVUps2bbR//36HmiZNmshmszncXnnlFYeao0ePqlWrVipSpIhKlSqlN954QxcvXnSoWb9+vWrXri0PDw9VrFhRs2fPvqSf6OholStXTp6engoJCdHWrVutnA4AAAAAAABgmaVALS4uTpGRkdq8ebNWr16tCxcuqHnz5kpPT3eoe/HFF3X8+HHzNnbsWHOf3W5Xq1atlJWVpU2bNumTTz7R7NmzFRUVZdYkJiaqVatWatq0qXbu3KnevXurW7duWrlypVnz5Zdfqm/fvho2bJh27NihmjVrKjw8XCdOnLjW1wIAAAAAAAC4IpthGMa1PvjkyZMqVaqU4uLi1LhxY0l/j1CrVauWJk6cmOdjli9frtatW+vYsWPy8/OTJE2fPl0DBgzQyZMn5e7urgEDBmjZsmXavXu3+bhnn31WKSkpWrFihSQpJCRE9erV09SpUyVJ2dnZCgwMVK9evTRw4MA8nzszM1OZmZnm/bS0NAUGBio1NVVeXl7X+jIAAAAAAAAgn0tLS5O3t/dV5UT/aQ611NRUSZKvr6/D9jlz5qhkyZK69957NWjQIJ07d87cFx8fr+rVq5thmiSFh4crLS1Ne/bsMWuaNWvmcMzw8HDFx8dLkrKysrR9+3aHGhcXFzVr1sysycuYMWPk7e1t3gIDA6/xzAEAAAAAAFBQFbrWB2ZnZ6t379564IEHdO+995rbn3vuOQUFBSkgIEAJCQkaMGCA9u/fr4ULF0qSkpKSHMI0Seb9pKSky9akpaUpIyNDf/31l+x2e541+/bt+9eeBw0apL59+5r3c0aoAQAAAAAAAFfrmgO1yMhI7d69Wz/88IPD9pdeesn8/+rVq6t06dJ66KGHdPDgQVWoUOHaO70OPDw85OHh4dQeAAAAAAAAkL9d0yWfPXv21NKlS7Vu3TqVKVPmsrUhISGSpAMHDkiS/P39lZyc7FCTc9/f3/+yNV5eXipcuLBKliwpV1fXPGtyjgEAAAAAAADcCJYCNcMw1LNnTy1atEhr165V+fLlr/iYnTt3SpJKly4tSQoNDdWuXbscVuNcvXq1vLy8VLVqVbNmzZo1DsdZvXq1QkNDJUnu7u6qU6eOQ012drbWrFlj1gAAAAAAAAA3gqVLPiMjIxUbG6slS5aoePHi5pxn3t7eKly4sA4ePKjY2Fi1bNlSJUqUUEJCgvr06aPGjRurRo0akqTmzZuratWq6tixo8aOHaukpCQNHTpUkZGR5uWYr7zyiqZOnar+/furS5cuWrt2rebNm6dly5aZvfTt21cRERGqW7eu6tevr4kTJyo9PV2dO3e+Xq8NAAAAAAAAcAmbYRjGVRfbbHlunzVrljp16qTffvtNzz//vHbv3q309HQFBgbqiSee0NChQx2WGz1y5Ii6d++u9evXq2jRooqIiNA777yjQoX+L99bv369+vTpo71796pMmTJ688031alTJ4fnnTp1qsaNG6ekpCTVqlVLkydPNi8xvRpWlkMFAAAAAADA7ctKTmQpULvdEKgBAAAAAABAspYTXdOiBAAAAAAAAEBBRaAGAAAAAAAAWECgBgAAAAAAAFhAoAYAAAAAAABYQKAGAAAAAAAAWECgBgAAAAAAAFhAoAYAAAAAAABYQKAGAAAAAAAAWECgBgAAAAAAAFhAoAYAAAAAAABYQKAGAAAAAAAAWECgBgAAAAAAAFhAoAYAAAAAAABYQKAGAAAAAAAAWECgBgAAAAAAAFhAoAYAAAAAAABYQKAGAAAAAAAAWECgBgAAAAAAAFhAoAYAAAAAAABYQKAGAAAAAAAAWECgBgAAAAAAAFhAoAYAAAAAAABYQKAGAAAAAAAAWECgBgAAAAAAAFhAoAYAAAAAAABYQKAGAAAAAAAAWECgBgAAAAAAAFhAoAYAAAAAAABYQKAGAAAAAAAAWECgBgAAAAAAAFhAoAYAAAAAAABYQKAGAAAAAAAAWECgBgAAAAAAAFhAoAYAAAAAAABYQKAGAAAAAAAAWECgBgAAAAAAAFhAoAYAAAAAAABYQKAGAAAAAAAAWECgBgAAAAAAAFhAoAYAAAAAAABYQKAGAAAAAAAAWECgBgAAAAAAAFhAoAYAAAAAAABYQKAGAAAAAAAAWECgBgAAAAAAAFhAoAYAAAAAAABYQKAGAAAAAAAAWECgBgAAAAAAAFhAoAYAAAAAAABYQKAGAAAAAAAAWECgBgAAAAAAAFhgKVAbM2aM6tWrp+LFi6tUqVJq06aN9u/f71Bz/vx5RUZGqkSJEipWrJjatm2r5ORkh5qjR4+qVatWKlKkiEqVKqU33nhDFy9edKhZv369ateuLQ8PD1WsWFGzZ8++pJ/o6GiVK1dOnp6eCgkJ0datW62cDgAAAAAAAGCZpUAtLi5OkZGR2rx5s1avXq0LFy6oefPmSk9PN2v69Omjb775RvPnz1dcXJyOHTumJ5980txvt9vVqlUrZWVladOmTfrkk080e/ZsRUVFmTWJiYlq1aqVmjZtqp07d6p3797q1q2bVq5cadZ8+eWX6tu3r4YNG6YdO3aoZs2aCg8P14kTJ/7L6wEAAAAAAABcls0wDONaH3zy5EmVKlVKcXFxaty4sVJTU3XnnXcqNjZWTz31lCRp3759qlKliuLj49WgQQMtX75crVu31rFjx+Tn5ydJmj59ugYMGKCTJ0/K3d1dAwYM0LJly7R7927zuZ599lmlpKRoxYoVkqSQkBDVq1dPU6dOlSRlZ2crMDBQvXr10sCBA/PsNzMzU5mZmeb9tLQ0BQYGKjU1VV5eXtf6MgAAAAAAACCfS0tLk7e391XlRP9pDrXU1FRJkq+vryRp+/btunDhgpo1a2bWVK5cWWXLllV8fLwkKT4+XtWrVzfDNEkKDw9XWlqa9uzZY9bkPkZOTc4xsrKytH37docaFxcXNWvWzKzJy5gxY+Tt7W3eAgMD/8vpAwAAAAAAoAC65kAtOztbvXv31gMPPKB7771XkpSUlCR3d3f5+Pg41Pr5+SkpKcmsyR2m5ezP2Xe5mrS0NGVkZOjPP/+U3W7PsybnGHkZNGiQUlNTzdtvv/1m/cQBAAAAAABQoBW61gdGRkZq9+7d+uGHH65nPzeUh4eHPDw8nN0GAAAAAAAA8rFrGqHWs2dPLV26VOvWrVOZMmXM7f7+/srKylJKSopDfXJysvz9/c2af676mXP/SjVeXl4qXLiwSpYsKVdX1zxrco4BAAAAAAAA3AiWAjXDMNSzZ08tWrRIa9euVfny5R3216lTR25ublqzZo25bf/+/Tp69KhCQ0MlSaGhodq1a5fDapyrV6+Wl5eXqlatatbkPkZOTc4x3N3dVadOHYea7OxsrVmzxqwBAAAAAAAAbgRLl3xGRkYqNjZWS5YsUfHixc35yry9vVW4cGF5e3ura9eu6tu3r3x9feXl5aVevXopNDRUDRo0kCQ1b95cVatWVceOHTV27FglJSVp6NChioyMNC/HfOWVVzR16lT1799fXbp00dq1azVv3jwtW7bM7KVv376KiIhQ3bp1Vb9+fU2cOFHp6enq3Lnz9XptAAAAAAAAgEvYDMMwrrrYZstz+6xZs9SpUydJ0vnz59WvXz998cUXyszMVHh4uKZNm+ZwKeaRI0fUvXt3rV+/XkWLFlVERITeeecdFSr0f/ne+vXr1adPH+3du1dlypTRm2++aT5HjqlTp2rcuHFKSkpSrVq1NHnyZIWEhFz1yVtZDhUAAAAAAAC3Lys5kaVA7XZDoAYAAAAAAADJWk50TYsSAAAAAAAAAAUVgRoAAAAAAABgAYEaAAAAAAAAYAGBGgAAAAAAAGABgRoAAAAAAABgAYEaAAAAAAAAYAGBGgAAAAAAAGABgRoAAAAAAABgAYEaAAAAAAAAYAGBGgAAAAAAAGABgRoAAAAAAABgAYEaAAAAAAAAYAGBGgAAAAAAAGABgRoAAAAAAABgAYEaAAAAAAAAYAGBGgAAAAAAAGABgRoAAAAAAABgAYEaAAAAAAAAYAGBGgAAAAAAAGABgRoAAAAAAABgAYEaAAAAAAAAYAGBGgAAAAAAAGABgRoAAAAAAABgAYEaAAAAAAAAYAGBGgAAAAAAAGABgRoAAAAAAABgAYEaAAAAAAAAYAGBGgAAAAAAAGABgRoAAAAAAABgAYEaAAAAAAAAYAGBGgAAAAAAAGABgRoAAAAAAABgAYEaAAAAAAAAYAGBGgAAAAAAAGABgRoAAAAAAABgAYEaAAAAAAAAYAGBGgAAAAAAAGABgRoAAAAAAABgAYEaAAAAAAAAYAGBGgAAAAAAAGABgRoAAAAAAABgAYEaAAAAAAAAYAGBGgAAAAAAAGABgRoAAAAAAABgAYEaAAAAAAAAYAGBGgAAAAAAAGABgRoAAAAAAABgAYEaAAAAAAAAYAGBGgAAAAAAAGABgRoAAAAAAABgAYEaAAAAAAAAYIHlQG3Dhg169NFHFRAQIJvNpsWLFzvs79Spk2w2m8PtkUcecag5ffq0OnToIC8vL/n4+Khr1646e/asQ01CQoIaNWokT09PBQYGauzYsZf0Mn/+fFWuXFmenp6qXr26vv32W6unAwAAAAAAAFhiOVBLT09XzZo1FR0d/a81jzzyiI4fP27evvjiC4f9HTp00J49e7R69WotXbpUGzZs0EsvvWTuT0tLU/PmzRUUFKTt27dr3LhxGj58uD744AOzZtOmTWrfvr26du2qn376SW3atFGbNm20e/duq6cEAAAAAAAAXDWbYRjGNT/YZtOiRYvUpk0bc1unTp2UkpJyyci1HD///LOqVq2qbdu2qW7dupKkFStWqGXLlvr9998VEBCgmJgYDRkyRElJSXJ3d5ckDRw4UIsXL9a+ffskSc8884zS09O1dOlS89gNGjRQrVq1NH369DyfOzMzU5mZmeb9tLQ0BQYGKjU1VV5eXtf6MgAAAAAAACCfS0tLk7e391XlRDdkDrX169erVKlSqlSpkrp3765Tp06Z++Lj4+Xj42OGaZLUrFkzubi4aMuWLWZN48aNzTBNksLDw7V//3799ddfZk2zZs0cnjc8PFzx8fH/2teYMWPk7e1t3gIDA6/L+QIAAAAAAKDguO6B2iOPPKJPP/1Ua9as0bvvvqu4uDi1aNFCdrtdkpSUlKRSpUo5PKZQoULy9fVVUlKSWePn5+dQk3P/SjU5+/MyaNAgpaammrfffvvtv50sAAAAAAAACpxC1/uAzz77rPn/1atXV40aNVShQgWtX79eDz300PV+Oks8PDzk4eHh1B4AAAAAAACQv92QSz5zCw4OVsmSJXXgwAFJkr+/v06cOOFQc/HiRZ0+fVr+/v5mTXJyskNNzv0r1eTsBwAAAAAAAG6EGx6o/f777zp16pRKly4tSQoNDVVKSoq2b99u1qxdu1bZ2dkKCQkxazZs2KALFy6YNatXr1alSpV0xx13mDVr1qxxeK7Vq1crNDT0Rp8SAAAAAAAACjDLgdrZs2e1c+dO7dy5U5KUmJionTt36ujRozp79qzeeOMNbd68WYcPH9aaNWv0+OOPq2LFigoPD5ckValSRY888ohefPFFbd26VRs3blTPnj317LPPKiAgQJL03HPPyd3dXV27dtWePXv05ZdfatKkSerbt6/Zx2uvvaYVK1Zo/Pjx2rdvn4YPH64ff/xRPXv2vA4vCwAAAAAAAJA3m2EYhpUHrF+/Xk2bNr1ke0REhGJiYtSmTRv99NNPSklJUUBAgJo3b6633nrLYQGB06dPq2fPnvrmm2/k4uKitm3bavLkySpWrJhZk5CQoMjISG3btk0lS5ZUr169NGDAAIfnnD9/voYOHarDhw/r7rvv1tixY9WyZcurPhcry6ECAAAAAADg9mUlJ7IcqN1OCNQAAAAAAAAgWcuJbvgcagAAAAAAAMDthEANAAAAAAAAsIBADQAAAAAAALCAQA0AAAAAAACwgEANAAAAAAAAsIBADQAAAAAAALCAQA0AAAAAAACwgEANAAAAAAAAsIBADQAAAAAAALCAQA0AAAAAAACwgEANAAAAAAAAsIBADQAAAAAAALCAQA0AAAAAAACwgEANAAAAAAAAsIBADQAAAAAAALCAQA0AAAAAAACwgEANAAAAAAAAsIBADQAAAAAAALCAQA0AAAAAAACwgEANAAAAAAAAsIBADQAAAAAAALCAQA0AAAAAAACwgEANAAAAAAAAsIBADQAAAAAAALCAQA0AAAAAAACwgEANAAAAAAAAsIBADQAAAAAAALCAQA0AAAAAAACwgEANAAAAAAAAsIBADQAAAAAAALCAQA0AAAAAAACwgEANAAAAAAAAsIBADQAAAAAAALCAQA0AAAAAAACwgEANAAAAAAAAsIBADQAAAAAAALCAQA0AAAAAAACwgEANAAAAAAAAsIBADQAAAAAAALCAQA0AAAAAAACwgEANAAAAAAAAsIBADQAAAAAAALCAQA0AAAAAAACwgEANAAAAAAAAsIBADQAAAAAAALCAQA0AAAAAAACwgEANAAAAAAAAsIBADQAAAAAAALCAQA0AAAAAAACwgEANAAAAAAAAsIBADQAAAAAAALDAcqC2YcMGPfroowoICJDNZtPixYsd9huGoaioKJUuXVqFCxdWs2bN9OuvvzrUnD59Wh06dJCXl5d8fHzUtWtXnT171qEmISFBjRo1kqenpwIDAzV27NhLepk/f74q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paWpatWqmjBhgjk39/bt2+Xi4nLJvNyzZ8/Wvn371KxZM2ecQoHENwTgCnL+YduzZ4/c3d21dOlSzZ8/X3PnzlXbtm01ZMgQff3112Z9WFiYpkyZogkTJjirZQByDMNnzZqlzz//XJ06ddITTzyho0ePasSIEcrIyJD0f6HaxIkTdejQIRYeACyqUqWKTp48qTfffFMbN25Udna2+vfvr9GjR+u1115TTEyMQ33OfEsA8od58+YpICBAX331lUqUKGFuf/TRRzVkyBAlJiZqypQp2rBhwyWP5ZdSwJXlvE9ef/11derUSePGjVP37t0VHh6u+fPnKzg4WGPHjlWjRo302GOPKT4+/pKrn+rUqaPt27fr7rvvdsYpFEgsSgBchZUrV6pFixYKCAjQN998o/vuu0+StH37dk2ePFnbt2/XO++8o9atWzs8Lq9huwBurjVr1ig+Pt6cg0KSVqxYoSeffFIvvPCCJkyYYE6OnntOCt6/gDV//fWXwsLCVLRoUY0dO1YNGzaUzWbTe++9p/79++uDDz5Qt27dnN0mgGtw+PBhDR06VHPnztU333yjFi1a6MKFC3Jzc5MkLV26VK+++qpeeOEFDR8+3LnNAvnUnDlz1K9fPy1fvlx33323srKyFBERoZSUFA0ZMkSPPPKIdu3apT59+qhw4cL65ptvzMfmXjwPNw+BGnAVfvnlF02fPl3R0dH6+OOP1aFDB3Pfjh07FB0drSVLlmjx4sVq2LChEzsFkMMwDB07dsycv2XMmDEaMGCAuX/FihVq27atIiIi9N5776lIkSLOahXIl7Zv367ixYvrnnvuMbedPn1aYWFh8vDw0MSJE3X//ffLxcVFkyZNUvPmzVWlShUndgzgavzbitbJycnq2rWrNm/erA0bNqhq1aoOC/b88MMPCg0NZc5g4Cr985e3w4cP1/fff69Vq1ZJ+nv+7VOnTqlNmzYqVqyYli9fLklKTExUUFAQv/i9BRCoAf/wb6NSjhw5otGjR+uzzz5TbGysw4IDW7Zs0XfffaeBAwfyIQJworx+O7djxw499NBDqlevnmbMmKHy5cub+3JGn7777rt64403bna7QL5kGIYyMzMVEBCgqlWratasWQ6Xl/z111+qVq2aatSooddff10PPvggH/qBfCL3z9HY2Fj98ssvstvtatiwocLDw3X69Gl17NhR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oUMH9uzZY76WOSFVp04d6tevz6ZNm/D19eXu3bucP38ef39/SpUqla0ytoiIyNMooSYikgPUrl2bXLlyERwczMWLF83XMwcGSUlJ/PTTT7z++uvZVqOpCpmIiLysihQpgslkYtasWezfvx94PCFlNBrJnz8/06dPp3r16kRERFC4cGHc3Ny4ceMGmzZtMk9KiYiI/BYVJRARySFWr15Nr169cHd3Z+TIkdSqVQuAq1ev4uXlRVJSErGxsVqRJiIi8n/i4+Px9vbGZDIxceJEGjRoAEBaWhq2trakpqaSmprK4sWLeffdd6levTrW1tYqQCAiIn9ICTURkRzCaDQSGhrKoEGDKFq0KNWqVSMjI4O7d++SkZHB/v37sbW1VQECERGRLLIm1SZMmEDDhg2Bx6u8r127hpeXF6VKlSIgIABAz1EREXkmSqiJiOQwcXFxhISEcObMGUqUKEGtWrUYMGCAZtRFRER+w9NWqt24cQMPDw+uXLnC6dOnsbW1tXSYIiKSgyihJiLyD6EZdRERkd+WmVQzGAwMHDiQhQsXcvnyZY4fP26uiq1JKREReVY6rVpEJAd62lyIkmkiIiK/rXz58vj7+2MwGPjggw+UTBMRkf+KVqiJiIiIiMhL44cffiAgIIB58+ZhY2OjZJqIiPwpSqiJiIiIiMhLSck0ERH5s5RQExEREREREREReQ46Q01EREREREREROQ5KKEmIiIiIiIiIiLyHJRQExEREREREREReQ5KqImIiIiIiIiIiDwHJdRERERERERERESegxJqIiIiIiIiIiIiz0EJNRERERERERERkeeghJqIiIiIiIiIiMhzUEJNRERERERERETkOfw/yWG60sQqz2sAAAAASUVORK5CYII="/>
</div>
</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell" id="cell-id=69c96af0-f524-4b2c-b306-d97c4f78f39e">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In [10]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
<div class="cm-editor cm-s-jupyter">
<div class="highlight hl-ipython3"><pre><span></span><span class="c1"># Engineer features</span>
<span class="n">X_train_engi</span> <span class="o">=</span> <span class="n">engineer_features</span><span class="p">(</span><span class="n">X_train</span><span class="p">)</span>
<span class="n">X_val_engi</span> <span class="o">=</span> <span class="n">engineer_features</span><span class="p">(</span><span class="n">X_val</span><span class="p">)</span>
<span class="n">X_test_engi</span> <span class="o">=</span> <span class="n">engineer_features</span><span class="p">(</span><span class="n">X_test</span><span class="p">)</span>
</pre></div>
</div>
</div>
</div>
</div>
<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>
<div class="jp-OutputArea jp-Cell-outputArea">
<div class="jp-OutputArea-child">
<div class="jp-OutputPrompt jp-OutputArea-prompt"></div>
<div class="jp-RenderedText jp-OutputArea-output" data-mime-type="text/plain" tabindex="0">
<pre>Engineer features
Engineer features
Engineer features
</pre>
</div>
</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell" id="cell-id=e4b93580-1ceb-4f5a-819a-caed5251f630">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In [11]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
<div class="cm-editor cm-s-jupyter">
<div class="highlight hl-ipython3"><pre><span></span><span class="c1"># Drop columns with missing</span>
<span class="n">df_drop</span><span class="p">,</span> <span class="n">hm_cols_to_drop</span> <span class="o">=</span> <span class="n">drop_high_missing_cols</span><span class="p">(</span><span class="n">X_train_engi</span><span class="p">,</span> <span class="n">threshold</span><span class="o">=</span><span class="mf">0.3</span><span class="p">)</span>
</pre></div>
</div>
</div>
</div>
</div>
<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>
<div class="jp-OutputArea jp-Cell-outputArea">
<div class="jp-OutputArea-child">
<div class="jp-OutputPrompt jp-OutputArea-prompt"></div>
<div class="jp-RenderedText jp-OutputArea-output" data-mime-type="text/plain" tabindex="0">
<pre>Dropping 0 columns at missing threshold &gt;30%
</pre>
</div>
</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell" id="cell-id=0b581d0e-9bf5-480b-8bec-4794558a02d0">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In [12]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
<div class="cm-editor cm-s-jupyter">
<div class="highlight hl-ipython3"><pre><span></span><span class="c1"># Drop correlated features here</span>
<span class="n">df_corr</span><span class="p">,</span> <span class="n">corr_cols_to_drop</span> <span class="o">=</span> <span class="n">drop_correlated</span><span class="p">(</span><span class="n">df_drop</span><span class="p">,</span> <span class="n">threshold</span><span class="o">=</span><span class="mf">0.8</span><span class="p">)</span>
</pre></div>
</div>
</div>
</div>
</div>
<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>
<div class="jp-OutputArea jp-Cell-outputArea">
<div class="jp-OutputArea-child">
<div class="jp-OutputPrompt jp-OutputArea-prompt"></div>
<div class="jp-RenderedText jp-OutputArea-output" data-mime-type="text/plain" tabindex="0">
<pre>Dropping 0 highly correlated features
</pre>
</div>
</div>
<div class="jp-OutputArea-child">
<div class="jp-OutputPrompt jp-OutputArea-prompt"></div>
<div class="jp-RenderedImage jp-OutputArea-output" tabindex="0">
<img alt="No description has been provided for this image" class="" src="data:image/png;base64,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WVw+lysfYbgsPDxcH330Ub4XM9aoUcOp3+Vaf/vtN6c/N9fSpUsXlS1bVm+99ZbS09PVrFkzhYaGKisrK1/fvLw8zZkzR/PmzdMPP/zg9P6DihUrFup4klStWrUivQRw5syZeu+995Senq4333xTVapUKfS+AADrIAQAAJQIl+82v/766woICMi33cPj//8nrU+fPvr666/1zDPPqFGjRvLx8VFeXp46d+58zbvWlxmGUWD7lb/M/dmf366fl5cnwzD04Ycfyt3dPV//az3z7uvrq6pVq+q77767bq1Xulrd16v1WtsuX68ZM2ZcdRm7q53LiRMn1KZNG/n6+mrixIkKCQmRl5eXtm7dqpEjRxbqZ3EzFHT9JTne3VAYNptNvXr10tKlS7V//36nl0P+2dSpUzV27Fg98MADmjRpkipUqCA3Nzc99dRTRTrna/2cCrJt2zZH6LVjxw7179+/SPsDAKyBEAAAUCKEhIRIkqpUqaIOHTpctd9vv/2mlJQUJSUlady4cY72y3e0r3S1X5ov3ym+8m3wkvLdUb9evXa7XbVq1VJYWFih97usa9eueuWVV7RhwwZFR0dfs2/NmjWVl5enPXv2KDw83NF++PBhnThxQjVr1izy8S+7fN19fX2ved0LkpqaqmPHjmnFihW64447HO0//PBDvr6FDTAun0tmZma+bbt27VKlSpVueHnGqxkwYIAWL14sNzc39evX76r93n77bbVr106LFi1yaj9x4oTjBZRS4c+5MM6cOaNBgwYpIiJCLVu21PTp09WzZ0/HCgQAAFzGOwEAACVCp06d5Ovrq6lTp+rChQv5tl9+o//lu75/vss7e/bsfPtc/mXxz7/s+/r6qlKlSvriiy+c2ufNm1foenv16iV3d3clJSXlq8VutzstV1iQZ599VmXKlNGQIUN0+PDhfNv37dunOXPmSPpjqrqU/xxfeOEFSVJsbGyh6/6zpk2bKiQkRDNnzlROTk6+7ddaSaGgn8X58+cLvI5lypQp1OMBgYGBatSokZYuXer0c/vuu+/08ccfO67FX6Fdu3aaNGmS5s6dW+BslMvc3d3z/cyXL1+uX375xantan/+imPkyJHKzs7W0qVL9cILLygoKEgDBw7Mt0QkAADMBAAAlAi+vr6aP3++7r//fjVp0kT9+vVT5cqVlZ2drdWrVysmJkZz586Vr6+v7rjjDk2fPl0XLlxQtWrV9PHHHxd497lp06aS/liirl+/fipVqpS6devm+OX7ueee05AhQxQVFaUvvvhCu3fvLnS9ISEhmjx5shITE5WVlaUePXqobNmy+uGHH/Tuu+/qwQcfVEJCwjX3f/PNN9W3b1+Fh4crLi5O9evX1/nz5/X1119r+fLlio+PlyQ1bNhQAwcO1CuvvOKYgp+WlqalS5eqR48eateuXdEu9hXc3Nz06quv6q677lK9evU0aNAgVatWTb/88os+++wz+fr66v333y9w35YtW6p8+fIaOHCgnnjiCRmGoddff73AafhNmzbVW2+9peHDh6tZs2by8fFRt27dChx3xowZuuuuuxQdHa3Bgwc7lggsV67cNafp3yg3NzeNGTPmuv26du2qiRMnatCgQWrZsqV27NihN954Q8HBwU79QkJC5Ofnp5dffllly5ZVmTJl1KJFi2u+s6Egn376qebNm6fx48c7lixcsmSJ2rZtq7Fjx2r69OlFGg8A8D/OtHUJAAC4wuXl5TZt2nTNfp999pm9U6dO9nLlytm9vLzsISEh9vj4ePvmzZsdfX7++Wd7z5497X5+fvZy5crZe/fubT9w4EC+5f3sdrt90qRJ9mrVqtnd3Nyclrc7e/asffDgwfZy5crZy5Yta+/Tp4/9yJEjV10i8OjRowXW+84779hbtWplL1OmjL1MmTL2unXr2h977DF7ZmZmoa7L7t277UOHDrUHBQXZPT097WXLlrXHxMTY//WvfzktsXfhwgV7UlKSvVatWvZSpUrZq1evbk9MTHTqY7f/sURgbGxsgddVkn358uUF1rFt2zZ7r1697BUrVrTbbDZ7zZo17X369LGnpKQ4+hS0ROD69evtt99+u7106dL2qlWr2p999lnHcnafffaZo19OTo59wIABdj8/P7skx3KBV1uu8ZNPPrHHxMTYS5cubff19bV369bNvnPnTqc+V/vZFFRnQa5cIvBqrrZE4IgRI+yBgYH20qVL22NiYuwbNmwocGm/9957zx4REWH38PBwOs82bdrY69WrV+Axrxzn1KlT9po1a9qbNGliv3DhglO/p59+2u7m5mbfsGHDNc8BAGAtht1ehLfiAAAAAACAEot3AgAAAAAAYBGEAAAAAAAAWAQhAAAAAAAAFkEIAAAAAACACb744gt169ZNVatW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FkEIAAAAAACARRACAAAAAABgEYQAAAAAAABYBCEAAAAAAAAWQQhQwiUnJ8vPz8/sMgAAAADgljHcDJf8lASWDgHi4+NlGIYMw1CpUqXk7++vjh07avHixcrLyzO7PNOkpqbKMAydOHHC7FIAAAAAADeRpUMASercubMOHjyorKwsffjhh2rXrp2efPJJde3aVRcvXjS7PAAAAAAAbhrLhwA2m00BAQGqVq2amjRpon/84x9677339OGHHyo5OVmSdOLECQ0ZMkSVK1eWr6+v7rzzTm3fvt0xxoQJE9SoUSMtWLBA1atXl7e3t/r06aOTJ086HevVV19VeHi4vLy8VLduXc2bN8+xLSsrS4ZhaMWKFWrXrp28vb3VsGFDbdiwwWmM5ORk1ahRQ97e3urZs6eOHTuW75zee+89NWnSRF5eXgoODlZSUpJToGEYhl599VX17NlT3t7eql27tlatWuWoo127dpKk8uXLyzAMxcfHS5LefvttNWjQQKVLl1bFihXVoUMHnTlzpvgXHwAAAABwS1k+BCjInXfeqYYNG2rFihWSpN69e+vIkSP68MMPtWXLFjVp0kTt27fX8ePHHfvs3btX//nPf/T+++9r7dq12rZtmx599FHH9jfeeEPjxo3TlClTlJGRoalTp2rs2LFaunSp07FHjx6thIQEpaenKywsTP3793f8Ar9x40YNHjxYw4YNU3p6utq1a6fJkyc77f/ll18qLi5OTz75pHbu3KkFCxYoOTlZU6ZMceqXlJSkPn366Ntvv1WXLl1077336vjx46pevbreeecdSVJmZqYOHjyoOXPm6ODBg+rfv78eeOABZWRkKDU1Vb169ZLdbr95Fx4AAAAA8JciBLiKunXrKisrS1999ZXS0tK0fPlyRUVFqXbt2po5c6b8/Pz09ttvO/qfO3dOr732mho1aqQ77rhD//rXv7Rs2TIdOnRIkjR+/HjNmjVLvXr1Uq1atdSrVy89/fTTWrBggdNxExISFBsbq7CwMCUlJenHH3/U3r17JUlz5sxR586d9eyzzyosLExPPPGEOnXq5LR/UlKSRo0apYEDByo4OFgdO3bUpEmT8h0nPj5e/fv3V2hoqKZOnaqcnBylpaXJ3d1dFSpUkCRVqVJFAQEBKleunA4ePKiLFy+qV69eCgoKUoMGDfToo4/Kx8fnpl97AAAAAMBf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AAAAAAAAiyAEAAAAAADAIlgdAAAAAABQohhuhtkllFjMBAAAAAAAwCIIAQAAAAAAsAhCAAAAAAAALIIQAAAAAAAAiyAEAAAAAADAIlgdAAAAAABQshjczy4urhwAAAAAABZBCAAAAAAAgEUQAgAAAAAAYBGEAAAAAAAAWAQhAAAAAAAAFsHqAAAAAACAksXNMLuCEouZAAAAAAAAWAQhAAAAAAAAFkEIAAAAAACARRACAAAAAABgEf/zIcCECRPUqFEjx/fx8fHq0aOHafUAAAAAAGAW00OADRs2yN3dXbGxsbfkeHPmzFFycvItOdZlbdu21VNPPeXUlpWVJcMwlJ6efktrAQAAAICSzjDcXPJTEphe5aJFi/T444/riy++0IEDB/7y45UrV05+fn5/+XEAAAAAAHA1poYAOTk5euutt/TII48oNjbW6Q59amqqDMPQ6tWrFRkZKS8vL91+++367rvvHH2Sk5Pl5+enlStXqnbt2vLy8lKnTp30008/XfWYf34cIC8vT9OnT1doaKhsNptq1KihKVOmOLaPHDlSYWFh8vb2VnBwsMaOHasLFy44tl9+3OD1119XUFCQypUrp379+un06dOO433++eeaM2eODMOQYRjKysrKV9fl801JSVFUVJS8vb3VsmVLZWZmOvV7//331axZM3l5ealSpUrq2bOnY9tvv/2muLg4lS9fXt7e3rrrrru0Z8+efNfrgw8+UJ06deTt7a177rlHZ8+e1dKlSxUUFKTy5cvriSee0KVLlxz75ebmKiEhQdWqVVOZMmXUokULpaamXvUaAwAAAABck6khwH/+8x/VrVtXderU0X333afFixfLbrc79XnmmWc0a9Ysbdq0SZUrV1a3bt2cfgk/e/aspkyZotdee03r16/XiRMn1K9fv0LXkJiYqOeee05jx47Vzp079eabb8rf39+xvWzZskpOTtbOnTs1Z84cLVy4UC+++KLTGPv27dPKlSv1wQcf6IMPPtDnn3+u5557TtIfjx9ER0dr6NChOnjwoA4ePKjq1atftZ7Ro0dr1qxZ2rx5szw8PPTAAw84tq1evVo9e/ZUly5dtG3bNqWkpKh58+aO7fHx8dq8ebNWrVqlDRs2yG63q0uXLvmu1z//+U8tW7ZMa9euVWpqqnr27Kk1a9ZozZo1ev3117VgwQK9/fbbjn2GDRumDRs2aNmyZfr222/Vu3dvde7c2SlgAAAAAAC4Pg8zD75o0SLdd999kqTOnTvr5MmT+vzzz9W2bVtHn/Hjx6tjx46SpKVLl+q2227Tu+++qz59+kiSLly4oLlz56pFixaOPuHh4UpLS3P6Bbkgp0+f1pw5czR37lwNHDhQkhQSEqJWrVo5+owZM8bxdVBQkBISErRs2TI9++yzjva8vDwlJyerbNmykqT7779fKSkpmjJlisqVKydPT095e3srICDgutdkypQpatOmjSRp1KhRio2N1blz5+Tl5aUpU6aoX79+SkpKcvRv2LChJGnPnj1atWqV1q9fr5YtW0qS3njjDVWvXl0rV65U7969Hddr/vz5CgkJkSTdc889ev3113X48GH5+PgoIiJC7dq102effaa+ffsqOztbS5YsUXZ2tqpWrSpJSkhI0Nq1a7VkyRJNnTr1uucEAAAAAHANpoUAmZmZSktL07vvvvtHIR4e6tu3rxYtWuQUAkRHRzu+rlChgurUqaOMjAxHm4eHh5o1a+b4vm7duvLz81NGRsZ1Q4CMjAzl5uaqffv2V+3z1ltv6Z///Kf27dunnJwcXbx4Ub6+vk59goKCHAGAJAUGBurIkSPXvgBXERkZ6TSOJB05ckQ1atRQenq6hg4detVz8fDwcIQhklSxYsV818vb29sRAEiSv7+/goKC5OPj49R2uf4dO3bo0qVLCgsLczpebm6uKlaseNXzyM3NVW5urlObzWa7an8AAAAAwF/PtBBg0aJFunjxouPusiTZ7XbZbDbNnTv3ltRQunTpa27fsGGD7r33XiUlJalTp04qV66cli1bplmzZjn1K1WqlNP3hmEoLy+vWDVdOZZhGJLkGOt69RZ1/MvHuFb9OTk5cnd315YtW+Tu7u7U78rg4M+mTZvmNGNB+mNWx5AB99xI+QAAAAAguRlmV1BimfJOgIsXL+q1117TrFmzlJ6e7vhs375dVatW1b///W9H32+++cbx9W+//abdu3crPDzcaazNmzc7vs/MzNSJEyec+lxN7dq1Vbp0aaWkpBS4/euvv1bNmjU1evRoRUVFqXbt2vrxxx+LfL6enp5OL9orrsjIyKvWGh4erosXL2rjxo2OtmPHjikzM1MRERHFPmbjxo116dIlHTlyRKGhoU6faz3ekJiYqJMnTzp9EhMTi10HAAAAAODG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VKZMGUVFRWnjxo2O7fPnz1dISIg8PT1Vp04dvf766/mu14IFC9S1a1d5e3srPDxcGzZs0N69e9W2bVuVKVNGLVu2dDpHSXrvvffUpEkTeXl5KTg4WElJSbp48eJVrzMAAAAA/GUMwzU/JUCRQ4Djx49r7dq1euyxx1SmTJl82/38/JSXl6fu3bvr+PHj+vzzz7Vu3Trt379fffv2dfTLyclRly5dlJKSom3btqlz587q1q2bsrOzncabMWOGGjZsqG3btmnUqFF68skntW7dOqc+Y8eO1d13363t27fr3nvvVb9+/ZSRkVGo80lPT1f79u0VERGhDRs26KuvvlK3bt106dIlSdKZM2c0fPhwbd68WSkpKXJzc1PPnj2Vl5fnNM7o0aOVkJCg9PR0hYWFqX///rp48aJatmyp2bNny9fXVwcPHtTBgweVkJBw1XpmzZqlqKgobdu2TY8++qgeeeQRZWZmOq5ZmzZt9Msvv2jVqlXavn27nn32WUct7777rp588kmNGDFC3333nR566CENGjRIn332mdMxJk2apLi4OKWnp6tu3boaMGCAHnroISUmJmrz5s2y2+0aNmyYo/+XX36puLg4Pfnkk9q5c6cWLFig5ORkTZkypVDXGAAAAADgGor8OMDevXtlt9tVt27dq/ZJSUnRjh079MMPP6h69eqSpNdee0316tXTpk2b1KxZMzVs2FANGzZ07DNp0iS9++67WrVqldMvoDExMRo1apQkKSwsTOvXr9eLL76ojh07Ovr07t1bQ4YMcYyzbt06/etf/9K8efOuez7Tp09XVFSUU9969eo5vr777rud+i9evFiVK1fWzp07Vb9+fUd7QkKCYmNjJUlJSUmqV6+e9u7dq7p166pcuXIyDKNQjyF06dJFjz76qCRp5MiRevHFF/XZZ5+pTp06evPNN3X06FFt2rRJFSpUkCSFhoY69p05c6bi4+Md+1+emTFz5ky1a9fO0W/QoEHq06eP4xjR0dEaO3asOnXqJEl68sknNWjQIEf/pKQkjRo1SgMHDpQkBQcHa9KkSXr22Wc1fvz4654TAAAAAMA1FHkmgN1uv26fjIwMVa9e3REASFJERIT8/Pwcd+hzcnKUkJCg8PBw+fn5ycfHRxkZGflmAkRHR+f7/s93+QvT52ouzwS4mj179qh///4KDg6Wr6+vgoKCJClf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+DEGDx6MGjVqoHfv3ujcuTMCAgIAAC1atMDIkSPRp08fmJubY/HixQCA0NBQNGrUCB4eHnB1dYVcLseBAwdQpkyZ9/5MLS0tDBkyBAUFBRg8ePAHeV5ERERERETvI5dpaeTHx0Am/yed9Ujyhg8fjkePHmHPnj3/6uufxv76Hyf6eBk3bI9byddFx9AINg41cDMlWXQMjWFr74DU9zT6lBI7e3uej7/Y2dvz78ZrbBxq8N/Ga+zs7fHo6lnRMTSCuVNT3E5OEh1DY1g71MT+MjVFx9AYXfOS+Lf0LzYONURH+NceXz4lOkKxKji3EB3hvT76LQJJPbKysnDp0iX873//+9cFACIiIiIiIhKLRYAPLC0t7Z1N965evYrq1aurMdG/0717d5w9exYjR45E+/btRcchIiIiIiKif4FFgA/MwsICcXFx7zz+MTh+/LjoCERERERERPT/xCLAB6ajowMHBwfRMYiIiIiIiIhYBCAiIiIiIqKPixwy0RE+Wh/HHgZERERERERE9P/GIgARERERERGRRLAIQERERERERCQRLAIQERERERERSQSLAEREREREREQSwd0BiIiIiIiI6KMil/F+9r/FM0dEREREREQkESwCEBEREREREUkEiwBEREREREREEsEiABEREREREZFEsAhAREREREREJBHcHYCIiIiIiIg+LjKZ6AQfLc4EICIiIiIiIpIIFgGIiIiIiIiIJIJFACIiIiIiIiKJYBGAiIiIiIiISCJYBCAiIiIiIiKSCO4OQERERERERB8VOe9n/2s8c0REREREREQSwSIAERERERERkUSwCEBEREREREQkESwCEBEREREREUkEiwBEREREREREEsHdAYiIiIiIiOijIpfJREf4aHEmABEREREREZFEyORyuVx0CCIiIiIiIqKSepB4QXSEYlWu3Uh0hPficgBSq7QbiaIjaIzqjrVxK/m66BgawcahBm6mJIuOoTFs7R1wLylBdAyNYVGzHs/HXyxq1tPYNz0iVK7dCHeuXxYdQ2NUq+GMP8/uFx1DI5Rr2hX3r10UHUNjVKnVgO85XmPjUAP7y9QUHUMjdM1LEh2BBOByACIiIiIiIiKJYBGAiIiIiIiISCK4HICIiIiIiIg+KnIZ72f/WzxzRERERERERBLBIgARERERERGRRLAIQERERERERCQRLAIQERERERERSQSLAEREREREREQSwd0BiIiIiIiI6KMih0x0hI8WZwIQERERERERSQSLAEREREREREQSwSIAERERERERkUSwCEBEREREREQkESwCEBEREREREUkEdwcgIiIiIiKij4pcxvvZ/xbPHBEREREREZFEsAhAREREREREJBEsAhARERERERFJBIsARERERERERBLBIgARERERERGRRHB3ACIiIiIiIvqoyGUy0RE+WpwJQERERERERCQRLAIQERERERERSQSLAEREREREREQSwSIAERERERERkUSwCEBEREREREQkEdwdgIiIiIiIiD4qcnB3gH+rVMwEGDJkCGQyGWQyGXR1deHg4IDAwEDk5+eLjlasOXPmoH79+h/ke8fExEBbWxtdu3b9IN+fiIiIiIiIPl6loggAAJ06dUJ6ejpu3LgBPz8/zJkzB0uWLFF53MuXLwWke0Uul3/wwkRwcDDGjRuHEydO4N69e8LzEBERERERkeYoNUUAPT09VKlSBdbW1hg1ahTc3d2xZ88eDBkyBD169MC8efNgYWGBmjVrAgAuXbqEtm3bwsDAABUqVMAXX3yB7Oxsxfcr+rqAgACYm5vD2NgYI0eOVCoiFBYWYsGCBbC1tYWBgQFcXFywbds2xfHjx49DJpPh4MGDaNSoEfT09PDzzz8jICAA8fHxitkLYWFhGDZsGDw8PJSeU15eHipVqoTg4OASnYPs7Gxs3rwZo0aNQteuXREWFqZ0vLg8UVFR730eBQUFGD58uOJ4zZo1sWLFihL/vyEiIiIiIiLNUGp7AhgYGODx48cAgIiICBgbG+PXX38FADx79gwdO3aEq6srzp07h4cPH8LHxwdjx45VunCOiIiAvr4+jh8/jlu3bmHo0KGoUKEC5s2bBwBYsGABfv75Z6xZswaOjo44ceIEBg4cCHNzc7Ru3VrxfaZOnYqlS5fCzs4O+vr68PPzw6FDh3D06FEAgImJCWrUqIFWrVohPT0dVatWBQDs27cPOTk56NOnT4me85YtW1CrVi3UrFkTAwcOxFdffYVp06ZBJlNeL/N6HjMzs/c+j8LCQlSrVg1bt25FhQoVcOrUKXzxxReoWrUqevfu/e/+BxEREREREZHalboigFwuR0REBA4fPoxx48bh0aNHMDQ0xLp166CrqwsA+Omnn/D8+XOEh4fD0NAQALBq1Sp0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/QGTaI6SruGVyWQoKCj4wGnoY5Gfn489e/bAy8sLALBw4UKMHDlSsetGafXkyRMkJydDLpfD0dERZmZmoiOp3dChQ0v0uNDQ0A+chOgVFgGIiIheY21tjQ0bNqBVq1Z4+fIlTE1NsXfvXkVDwEuXLqF169aSuetNRB+GsbEx4uLiYGdnJzoKEUmMjugAREREmqRLly6YOnUqFi1ahF27dqFs2bJK67wTEhJgb28vMKH6FRQUYOfOnUhMTAQAODk5oXv37orlAUT0z/E+HBGJwldvIiKi18ydOxdeXl5o3bo1jIyMsH79eujq6iqOh4SEoEOHDgITqteVK1fg6emJ+/fvo2bNmgCARYsWwdzcHHv37oWzs7PghGJERETg22+/VRRGateuja+++gru7u6CkxEREb0blwMQEREVIysrC0ZGRtDW1lYaz8jIgJGRkVJhoDRzdXWFubk51q9fr1jL++TJEwwZMgSPHj3CqVOnBCdUv++//x7jx4/HZ599BldXVwCvGgNu27YN3377LcaMGSM4IX0MypUrh/j4eC4HICK1YxGAiIiI3srAwADnz59HnTp1lMYvX76MJk2aIDc3V1AycapVq4apU6di7NixSuOrV6/G/PnzcffuXUHJ6GPCIgARiVKy9rZEREQkSTVq1MCDBw9Uxh8+fAgHBwcBicTLzMxEp06dVMY7dOiArKwsAYmIiIhKjkUAIiIieqsFCxbA19cX27Ztw507d3Dnzh1s27YNX331FRYtWoSnT58qPqTC09MTO3fuVBnfvXs3PDw8BCSij1HLli1hYGAgOgYRSRCXAxAREdFbaWn9fb9AJpMB+Lur+eufy2QyFBQUqD+gAN988w2WLl0KNzc3pZ4A0dHR8PPzg7GxseKxvr6+omKSQIWFhUhOTsbDhw9RWFiodKxVq1aCUhERvcIiABEREb1VZGTkW48lJCSgXr16is9bt26tjkjC2draluhxMpkMqampHzgNaZrTp0+jf//+uH37tso2gFIqlhGR5mIRgIiIiErszz//xMaNG7Fu3TpcuHCBFzREb6hfvz5q1KiBgIAAVK1aVTFjpoiJiYmgZEREr7AIQERERO914sQJBAcHY/v27bCwsICXlxd69eqFJk2aiI5GpFEMDQ0RHx8v2caZRKT5dEQHICIiIs10//59hIWFITg4GE+fPkXv3r3x4sUL7Nq1C05OTqLjCXXnzh3s2bMHaWlpePnypdKx5cuXC0pFmqBZs2ZITk5mEYCINBaLAERERKSiW7duOHHiBLp27YqgoCB06tQJ2traWLNmjehowkVERMDT0xN2dna4du0anJ2dcevWLcjlcjRs2FB0PBJs3Lhx8PPzw/3791G3bl2UKVNG6fjrfTSIiETgcgAiIiJSoaOjA19fX4waNQqOjo6K8TJlyiA+Pl7SMwGaNm2Kzp07IyAgAOXKlUN8fDwqVaqEAQMGoFOnThg1apToiCTQ6ztqFJHJZJLbRYOINBdnAhAREZGKqKgoBAcHo1GjRqhduzYGDRqEvn37io6lERITE7Fx40YAr4olubm5MDIyQmBgILp3784igMTdvHlTdAQiondiEYCIiIhUNG/eHM2bN0dQUBA2b96MkJAQTJw4EYWFhfj1119hZWWFcuXKiY4phKGhoaIPQNWqVZGSkoI6deoAAP744w+R0UgDWFtbi45ARPROXA5AREREJZKUlITg4GBs2LABmZmZaN++Pfbs2SM6ltr16NEDXbt2xYgRIzBp0iTs3r0bQ4YMwY4dO2BmZoajR4+Kjkga4OrVq8U2jvT09BSUiIjoFRYBiIiI6B8pKCjA3r17ERISIskiQGpqKrKzs1GvXj08e/YMfn5+OHXqFBwdHbF8+XLeCZa41NRU9OzZE5cuXVL0AgBe9QUAwJ4ARCQciwBERERERP+Rbt26QVtbG+vWrYOtrS3Onj2Lx48fw8/PD0uXLkXLli1FRyQiiWMRgIiIiOhfuHDhAhITEwEAderUQYMGDQQnIk1QsWJFHDt2DPXq1YOJiQnOnj2LmjVr4tixY/Dz88PFixdFRyQiiWNjQCIiIqJ/4OHDh+jbty+OHz8OU1NTAEBmZibatGmDTZs2wdzcXGxAEqqgoEDRNLNixYq4d+8eatasCWtrayQlJQlOR0QEqG5kSkRERERvNW7cOPz555+4cuUKMjIykJGRgcuXL+Pp06fw9fUVHY8Ec3Z2Rnx8PACgWbNmWLx4MaKjoxEYGAg7OzvB6YiIuByAiIiI6B8xMTHB0aNH0aRJE6Xxs2fPokOHDsjMzBQTjDTC4cOH8ezZM3h5eSE5ORkeHh64fv06KlSogM2bN6Nt27aiIxKRxHE5ABEREdE/UFhYiDJlyqiMlylTBoWFhQISkSbp2LGj4r8dHBxw7do1ZGRkwMzMTLFDABGRSJwJQERERPQPdO/eHZmZmdi4cSMsLCwAAHfv3sWAAQNgZmaGnTt3Ck5ImuLOnTsAgGrVqglOQkT0N/YEICIiIvoHVq1ahadPn8LGxgb29vawt7eHra0tnj59ipUrV4qOR4IVFhYiMDAQJiYmsLa2hrW1NUxNTTF37lzOFCEijcDlAERERET/gJWVFWJjY3H06FFcu3YNAFC7dm24u7sLTkaaYPr06QgODsbChQvh5uYGAIiKisKcOXPw/PlzzJs3T3BCIpI6LgcgIiIiKoFjx45h7NixOH36NIyNjZWOZWVloUWLFlizZg1atmwpKCFpAgsLC6xZswaenp5K47t378bo0aNx9+5dQcmIiF7hcgAiIiKiEggKCsKIESNUCgDAqx0DvvzySyxfvlxAMtIkGRkZqFWrlsp4rVq1kJGRISAREZEyFgGIiIiISiA+Ph6dOnV66/EOHTrgwoULakxEmsjFxQWrVq1SGV+1ahXq1asnIBERkTL2BCAiIiIqgQcPHhS7NWARHR0dPHr0SI2JSBMtXrwYXbt2xdGjR+Hq6goAiImJwe+//44DBw4ITkdExJkARERERCViaWmJy5cvv/V4QkICqlatqsZEpIlat26N69evo2fPnsjMzERmZia8vLxw5coVbNiwQXQ8IiI2BiQiIiIqiXHjxuH48eM4d+4c9PX1lY7l5uaiadOmaNOmDb777jtBCUmTxcfHo2HDhigoKBAdhYgkjkUAIiIiohJ48OABGjZsCG1tbYwdOxY1a9YEAFy7dg2rV69GQUEBYmNjUblyZcFJSROxCEBEmoI9AYiIiIhKoHLlyjh16hRGjRqFadOmoeg+ikwmQ8eOHbF69WoWAIiISONxJgARERHRP/TkyRMkJydDLpfD0dERZmZmoiORhuNMACLSFCwCEBERERH9P3l5eb3zeGZmJiIjI1kEICLhuByAiIiIiOj/ycTE5L3HBw8erKY0RERvx5kARERERERERBKhJToAEREREREREakHiwBEREREREREEsEiABEREREREZFEsAhAREREREREJBEsAhARERERERFJBIsARERERERERBLBIgARERERERGRRPwflsobOKprG6IAAAAASUVORK5CYII="/>
</div>
</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell" id="cell-id=e85b809c-09e4-4d23-be60-368a6459cdda">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In [13]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
<div class="cm-editor cm-s-jupyter">
<div class="highlight hl-ipython3"><pre><span></span><span class="c1"># Drop high card</span>
<span class="n">df_high</span><span class="p">,</span> <span class="n">hc_cols_to_drop</span> <span class="o">=</span> <span class="n">drop_high_card_cols</span><span class="p">(</span><span class="n">df_corr</span><span class="p">,</span> <span class="n">threshold</span><span class="o">=</span><span class="mi">50</span><span class="p">)</span>
</pre></div>
</div>
</div>
</div>
</div>
<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>
<div class="jp-OutputArea jp-Cell-outputArea">
<div class="jp-OutputArea-child">
<div class="jp-OutputPrompt jp-OutputArea-prompt"></div>
<div class="jp-RenderedText jp-OutputArea-output" data-mime-type="text/plain" tabindex="0">
<pre>Dropping 0 high-cardinality columns (&gt; 50 unique values)
</pre>
</div>
</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell" id="cell-id=42e5881e-dbf5-467b-8290-3f7d9c48e3ff">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In [14]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
<div class="cm-editor cm-s-jupyter">
<div class="highlight hl-ipython3"><pre><span></span><span class="c1"># Collapse rare categories on training data</span>
<span class="n">df_collapsed</span><span class="p">,</span> <span class="n">rare_maps</span> <span class="o">=</span> <span class="n">collapse_rare_categories</span><span class="p">(</span><span class="n">df_high</span><span class="p">,</span> <span class="n">min_freq</span><span class="o">=</span><span class="mf">0.005</span><span class="p">)</span>
</pre></div>
</div>
</div>
</div>
</div>
<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>
<div class="jp-OutputArea jp-Cell-outputArea">
<div class="jp-OutputArea-child">
<div class="jp-OutputPrompt jp-OutputArea-prompt"></div>
<div class="jp-RenderedText jp-OutputArea-output" data-mime-type="text/plain" tabindex="0">
<pre>Nothing to collapse
</pre>
</div>
</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell" id="cell-id=bf996eee-3c94-4dc1-9aeb-65c9b43e1c8e">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In [15]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
<div class="cm-editor cm-s-jupyter">
<div class="highlight hl-ipython3"><pre><span></span><span class="c1"># Impute and scale</span>
<span class="n">df_processed</span><span class="p">,</span> <span class="n">num_imputer</span><span class="p">,</span> <span class="n">cat_imputer</span><span class="p">,</span> <span class="n">robust_scaler</span><span class="p">,</span> <span class="n">std_scaler</span> <span class="o">=</span> <span class="n">impute_and_scale</span><span class="p">(</span><span class="n">df_collapsed</span><span class="p">,</span> <span class="n">threshold</span><span class="o">=</span><span class="mf">1.0</span><span class="p">)</span>
</pre></div>
</div>
</div>
</div>
</div>
<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>
<div class="jp-OutputArea jp-Cell-outputArea">
<div class="jp-OutputArea-child">
<div class="jp-OutputPrompt jp-OutputArea-prompt"></div>
<div class="jp-RenderedText jp-OutputArea-output" data-mime-type="text/plain" tabindex="0">
<pre>Imputed and scaled features
</pre>
</div>
</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell" id="cell-id=b7ebfce6-1f69-42dc-9fe6-1cdfe2e5c0f3">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In [16]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
<div class="cm-editor cm-s-jupyter">
<div class="highlight hl-ipython3"><pre><span></span><span class="c1"># Random Forest feature selection</span>
<span class="n">df_selected</span><span class="p">,</span> <span class="n">selected_features</span> <span class="o">=</span> <span class="n">select_features_rf</span><span class="p">(</span><span class="n">df_processed</span><span class="p">,</span> <span class="n">y_train</span><span class="p">,</span> <span class="n">threshold</span><span class="o">=</span><span class="mf">0.02</span><span class="p">,</span> <span class="n">top_n</span><span class="o">=</span><span class="mi">30</span><span class="p">)</span>
</pre></div>
</div>
</div>
</div>
</div>
<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>
<div class="jp-OutputArea jp-Cell-outputArea">
<div class="jp-OutputArea-child">
<div class="jp-OutputPrompt jp-OutputArea-prompt"></div>
<div class="jp-RenderedImage jp-OutputArea-output" tabindex="0">
<img alt="No description has been provided for this image" class="" src="data:image/png;base64,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8+bNWY7p7u5u/XnOnDkKDQ3VunXrtG3bNr344ouaMWOG9u7dmynk/p23t3emYHwncnounZycMv1hJD09XUWLFtX777+f5RgZf8CwWCz6+OOPtXfvXn322WfaunWr+vXrpzlz5mjv3r02xyMvf7f+Li0tTefOnctR38KFC8vR0VHSX/9tOnnyZKY+OflvU+HCheXk5JTr/7Y999xz2rBhg95//33r1ThZuXjxYqaHrwH4dyF0A0A+Vbp0ae3YsSPTA3x++OEH6/K/i4+PzzTGjz/+KFdX11xdFpwh43LKv89K1qxZU7t371Z6erpNENi3b59cXV3vOGRmKFeunGJiYtSyZctbzgx99tlnSk1N1fr1621m6P5++XmG7MbJmGG7dOmS9TJcKfMM7+3qTUlJuSdPsr6XTp06lWnW8ccff5T0f0+lL126tL799ttM5zK736+sZHdsjx49qh9//FErV65Unz59rO3bt2/P9b5kKFeunLZu3aoLFy5kO9tdrlw5GYahMmXK5Oh3sVq1aqpWrZomTJigb775RiEhIXrrrbc0bdq0bNepVKmSPvnkkzvej3uhXLly2rFjh0JCQnL0x5CGDRuqYcOGmj59uj744AM9/fTTWrVqlQYMGJDjbWb8PsTFxWVa9sMPP6hIkSJZznL/fd34+HiVLVvW2n7u3LlMf8A4ceJEjq+q+OKLL6xvI6hZs6a++OILJScn2zxMbd++fdbl2bGzs1O1atUyPQU/Y/2yZctmuuLo5ZdfVkREhObPn29zi0ZWfvnlF5tL4QH8+3BPNwDkU+3atVNaWpoWLlxo0z5v3jxZLBY99thjNu179uzR4cOHrd9PnDihdevW6dFHH73lTOIff/yR5WzW22+/LUnWp0JLf81+//777zZPP/7jjz+0evVqdezYMct7InOje/fuOnnypJYtW5Zp2bVr16z3mGfsz9/rTkpKUkRERKb13NzcbF57lqFcuXKSZHPf9ZUrVzK9Mut29e7Zs0dbt27NtOzSpUu6efNmjse6l27evGl9pZn015PplyxZIh8fH+t9/+3atdOZM2f04Ycf2qz33//+V+7u7mratOltt5MRsv55fLM6P4ZhaMGCBXe8T127dpVhGAoPD8+0LGM7TzzxhOzt7RUeHp7pd9owDJ0/f16SlJycnOncVKtWTXZ2drd97V1wcLAuXryY5b3X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9vr27aukpCRZLBZZLBZNnjw5y9oCAgL06quvql+/fipYsKBKlSqlpUuX2vT57bff1KtXLxUuXFhubm6qW7eu9u3bZ12+ePFilStXTo6OjqpYsaLefffdTMdryZIl6tChg1xdXRUUFKQ9e/bop59+UrNmzeTm5qZGjRrZ7KMkrVu3TrVr15azs7PKli2r8PBw3bx5M9vjDAAAAAC4g9B94cIFbdmyRS+88ILc3NwyLffy8lJ6ero6deqkCxcu6Msvv9T27dv1888/q0ePHtZ+KSkpateunXbu3KkjR46obdu26tixoxITE23GmzVrlmrUqKEjR45o7Nixeumll7R9+3abPq+88oq6du2qmJgYPf300+rZs6diY2NztD/R0dFq2bKlKleurD179ujrr79Wx44dlZaWJkm6cuWKRo4cqYMHD2rnzp2ys7NTly5dlJ6ebjPO+PHjFRYWpujoaAUGBqpXr166efOmGjVqpPnz58vDw0OnT5/W6dOnFRYWlm09c+bMUd26dXXkyBENGTJEzz//vOLi4qzHrGnTpjp58qTWr1+vmJgYjR492lrLp59+qpdeekmjRo3Sd999p0GDBqlv37764osvbLYxdepU9enTR9HR0apUqZKeeuopDRo0SOPGjdPBgwdlGIaGDh1q7b9792716dNHL730ko4dO6YlS5YoMjJS06dPz9ExBgAAAID8ymIYhpGbFfbv368GDRpozZo16tKlS5Z9tm/frscee0y//PKL/P39JUnHjh1TlSpVtH//ftWrVy/L9apWrarBgwdbA19AQICCgoK0efNma5+ePXsqOTlZmzZt+msHLBYNHjxYixcvtvZp2LChateurUWLFikhIUFlypTRkSNHVLNmTe3atUvNmzfXxYsX5eXlpaeeekqJiYn6+uuvc7T/f/zxh3x8fHT06FFVrVrVOv7bb7+t/v372+xrbGysKlWqpMjISA0fPjzTw9uaNWummjVrav78+db9bdKkiXV22jAM+fr6Kjw8XIMHD9bSpUsVFhamhIQEFS5cOFNtISEhqlKlis3sePfu3XXlyhVt3LjRerwmTJigqVOnSpL27t2r4OBgLV++XP369ZMkrVq1Sn379tW1a9ckSa1atVLLli01btw467jvvfeeRo8erVOnTmV5nFJTU5Wammr9npycLH9/f0366mc5uxfM0bEG8GAa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OmrFihVydXVVlSpVNGXKFL388suaOnWqbty4oVdffVU7duxQcHCwJKls2bL6+uuvtWTJEpvQPWXKFLVu3dr63d3dXQ4ODvL19bW2NWrUSBUrVtS7776r0aNHS5IiIiLUrVs3ubu752jfly9frt69e0v663L7pKQkffnll2rWrJlNv7/Xk5qaetv9KFCggMLDw63rlylTRnv27NFHH32UbehOTU1Vamqq9XtycnKO9gEAAAAAHiYPTejesGGD3N3ddePGDaWnp+upp57S5MmT9cILL6hatWrWwC1JsbGxqlGjhjVwS1JISIjS09MVFxdnDd01atSQq6urtU9wcLBSUlJ04sQJpaSk6OrVqzZhWvrrnvFatWrZtNWtWzdH+zBgwAAtXbpUo0eP1u+//67Nmzfr888/z9G6cXFx2r9/vz799FNJkoODg3r06KHly5dnCt1/r+enn37K0X68+eabWrFihRITE3Xt2jVdv379lg+DmzFjhk1QBwAAAID86KEJ3c2bN9fixYvl6Oio4sWLy8Hh/3bt7+H6Xsm4/3vjxo0qUaKEzTInJyeb7zndfp8+fTR27Fjt2bNH33zzjcqUKaMmTZrkaN3ly5fr5s2bKl68uLXNMAw5OTlp4cKF8vT0zLKenOzHqlWrFBYWpjlz5ig4OFgFCxbUrFmztG/fvmzrGTdunEaOHGn9npycLH9//xztCwAAAAA8LB6a0O3m5qby5cvnqG9QUJAiIyN15coVawCNioqSnZ2d9UFrkhQTE6Nr167JxcVFkrR37165u7vL399fhQsXlpOTkxITE20uJc8JR0dHpaWlZWr39vZW586dFRERoT179qhv3745Gu/mzZt65513NGfOHD366KM2yzp37qz//e9/Gjx4cJbrVq5c+bb7ERUVpUaNGmnIkCHWtuPHj9+yJicnp0x/fAAAAACA/OahCd258fTTT2vSpEl69tlnNXnyZJ07d07Dhg3TM888Y720XPrrEuv+/ftrwoQJSkhI0KRJkzR06FDZ2dmpYMGCCgsL04gRI5Senq7GjRsrKSlJUVFR8vDw0LPPPpvt9gMCAvTLL78oOjpaJUuWVMGCBa0BdcCAAerQoYPS0tJuOcbfbdiwQRcvXlT//v1tZrQlqWvXrlq+fHm2oTsn+1GhQgW988472rp1q8qUKaN3331XBw4cUJkyZXJUHwAA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"/>
</div>
</div>
<div class="jp-OutputArea-child">
<div class="jp-OutputPrompt jp-OutputArea-prompt"></div>
<div class="jp-RenderedText jp-OutputArea-output" data-mime-type="text/plain" tabindex="0">
<pre>Number of features before selection: 11
Selected 9 features (threshold=0.02)
</pre>
</div>
</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs" id="cell-id=9af7c210-987d-4713-9a63-8e76e6033b5e">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In [17]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
<div class="cm-editor cm-s-jupyter">
<div class="highlight hl-ipython3"><pre><span></span><span class="c1"># Process</span>
<span class="n">all_cols_to_drop</span> <span class="o">=</span> <span class="n">hm_cols_to_drop</span> <span class="o">+</span> <span class="n">hc_cols_to_drop</span> <span class="o">+</span> <span class="n">corr_cols_to_drop</span>
<span class="n">X_val</span> <span class="o">=</span> <span class="n">transform_val_test</span><span class="p">(</span><span class="n">X_val_engi</span><span class="p">,</span> <span class="n">all_cols_to_drop</span><span class="p">,</span> <span class="n">selected_features</span><span class="p">,</span> <span class="n">rare_maps</span><span class="p">,</span> <span class="n">num_imputer</span><span class="p">,</span> <span class="n">cat_imputer</span><span class="p">,</span> <span class="n">robust_scaler</span><span class="p">,</span> <span class="n">std_scaler</span><span class="p">)</span>
<span class="n">X_test</span> <span class="o">=</span> <span class="n">transform_val_test</span><span class="p">(</span><span class="n">X_test_engi</span><span class="p">,</span> <span class="n">all_cols_to_drop</span><span class="p">,</span> <span class="n">selected_features</span><span class="p">,</span> <span class="n">rare_maps</span><span class="p">,</span> <span class="n">num_imputer</span><span class="p">,</span> <span class="n">cat_imputer</span><span class="p">,</span> <span class="n">robust_scaler</span><span class="p">,</span> <span class="n">std_scaler</span><span class="p">)</span>
<span class="n">X_train</span> <span class="o">=</span> <span class="n">df_selected</span><span class="o">.</span><span class="n">copy</span><span class="p">()</span>
</pre></div>
</div>
</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell" id="cell-id=c565e000-f300-47e3-93f2-5205fd8e4e86">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In [18]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
<div class="cm-editor cm-s-jupyter">
<div class="highlight hl-ipython3"><pre><span></span><span class="c1">#summary</span>
<span class="nb">print</span><span class="p">(</span><span class="n">dataset_summary</span><span class="p">(</span><span class="n">X_train</span><span class="p">))</span>
</pre></div>
</div>
</div>
</div>
</div>
<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>
<div class="jp-OutputArea jp-Cell-outputArea">
<div class="jp-OutputArea-child">
<div class="jp-OutputPrompt jp-OutputArea-prompt"></div>
<div class="jp-RenderedText jp-OutputArea-output" data-mime-type="text/plain" tabindex="0">
<pre>Dataset shape: (243, 9)
Total rows: 243
Total duplicate rows: 0
                     dtype  non_null_count  missing_count  missing_%  \
Credit_History     float32             243              0        0.0   
ApplicantIncome    float32             243              0        0.0   
LoanAmount         float32             243              0        0.0   
CoapplicantIncome  float32             243              0        0.0   
Loan_Amount_Term   float32             243              0        0.0   
Property_Area       object             243              0        0.0   
Dependents          object             243              0        0.0   
Married             object             243              0        0.0   
Education           object             243              0        0.0   

                   unique_count  duplicates_in_dataset  
Credit_History                2                      0  
ApplicantIncome             216                      0  
LoanAmount                   89                      0  
CoapplicantIncome           121                      0  
Loan_Amount_Term              8                      0  
Property_Area                 3                      0  
Dependents                    4                      0  
Married                       2                      0  
Education                     2                      0  
</pre>
</div>
</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell" id="cell-id=98be230c-b9da-4bd8-9acf-cf3e640509aa">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In [19]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
<div class="cm-editor cm-s-jupyter">
<div class="highlight hl-ipython3"><pre><span></span><span class="c1"># Encode</span>
<span class="n">le</span> <span class="o">=</span> <span class="n">LabelEncoder</span><span class="p">()</span>
<span class="n">y_train_encoded</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">Series</span><span class="p">(</span><span class="n">le</span><span class="o">.</span><span class="n">fit_transform</span><span class="p">(</span><span class="n">y_train</span><span class="p">),</span> <span class="n">index</span><span class="o">=</span><span class="n">y_train</span><span class="o">.</span><span class="n">index</span><span class="p">)</span>
<span class="n">y_train</span> <span class="o">=</span> <span class="n">le</span><span class="o">.</span><span class="n">fit_transform</span><span class="p">(</span><span class="n">y_train</span><span class="p">)</span>  
<span class="n">y_val</span> <span class="o">=</span> <span class="n">le</span><span class="o">.</span><span class="n">transform</span><span class="p">(</span><span class="n">y_val</span><span class="p">)</span>       
<span class="n">y_test</span> <span class="o">=</span> <span class="n">le</span><span class="o">.</span><span class="n">transform</span><span class="p">(</span><span class="n">y_test</span><span class="p">)</span>    

<span class="n">cat_cols</span> <span class="o">=</span> <span class="n">X_train</span><span class="o">.</span><span class="n">select_dtypes</span><span class="p">(</span><span class="n">include</span><span class="o">=</span><span class="p">[</span><span class="s1">'object'</span><span class="p">,</span> <span class="s1">'category'</span><span class="p">])</span><span class="o">.</span><span class="n">columns</span><span class="o">.</span><span class="n">tolist</span><span class="p">()</span>

<span class="n">X_train</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">get_dummies</span><span class="p">(</span><span class="n">X_train</span><span class="p">,</span> <span class="n">columns</span><span class="o">=</span><span class="n">cat_cols</span><span class="p">,</span> <span class="n">drop_first</span><span class="o">=</span><span class="kc">False</span><span class="p">)</span>
<span class="n">X_val</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">get_dummies</span><span class="p">(</span><span class="n">X_val</span><span class="p">,</span> <span class="n">columns</span><span class="o">=</span><span class="n">cat_cols</span><span class="p">,</span> <span class="n">drop_first</span><span class="o">=</span><span class="kc">False</span><span class="p">)</span>
<span class="n">X_test</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">get_dummies</span><span class="p">(</span><span class="n">X_test</span><span class="p">,</span> <span class="n">columns</span><span class="o">=</span><span class="n">cat_cols</span><span class="p">,</span> <span class="n">drop_first</span><span class="o">=</span><span class="kc">False</span><span class="p">)</span>
<span class="n">X_val</span> <span class="o">=</span> <span class="n">X_val</span><span class="o">.</span><span class="n">reindex</span><span class="p">(</span><span class="n">columns</span><span class="o">=</span><span class="n">X_train</span><span class="o">.</span><span class="n">columns</span><span class="p">,</span> <span class="n">fill_value</span><span class="o">=</span><span class="mi">0</span><span class="p">)</span>
<span class="n">X_test</span> <span class="o">=</span> <span class="n">X_test</span><span class="o">.</span><span class="n">reindex</span><span class="p">(</span><span class="n">columns</span><span class="o">=</span><span class="n">X_train</span><span class="o">.</span><span class="n">columns</span><span class="p">,</span> <span class="n">fill_value</span><span class="o">=</span><span class="mi">0</span><span class="p">)</span>

<span class="n">X_train</span> <span class="o">=</span> <span class="n">X_train</span><span class="o">.</span><span class="n">astype</span><span class="p">(</span><span class="s1">'float32'</span><span class="p">)</span>
<span class="n">X_val</span> <span class="o">=</span> <span class="n">X_val</span><span class="o">.</span><span class="n">astype</span><span class="p">(</span><span class="s1">'float32'</span><span class="p">)</span>
<span class="n">X_test</span> <span class="o">=</span> <span class="n">X_test</span><span class="o">.</span><span class="n">astype</span><span class="p">(</span><span class="s1">'float32'</span><span class="p">)</span>

<span class="nb">print</span><span class="p">(</span><span class="s2">"Train shape:"</span><span class="p">,</span> <span class="n">X_train</span><span class="o">.</span><span class="n">shape</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="s2">"Val shape:  "</span><span class="p">,</span> <span class="n">X_val</span><span class="o">.</span><span class="n">shape</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="s2">"Test shape: "</span><span class="p">,</span> <span class="n">X_test</span><span class="o">.</span><span class="n">shape</span><span class="p">)</span>

<span class="n">missing_val_cols</span> <span class="o">=</span> <span class="nb">set</span><span class="p">(</span><span class="n">X_train</span><span class="o">.</span><span class="n">columns</span><span class="p">)</span> <span class="o">-</span> <span class="nb">set</span><span class="p">(</span><span class="n">X_val</span><span class="o">.</span><span class="n">columns</span><span class="p">)</span>
<span class="n">missing_test_cols</span> <span class="o">=</span> <span class="nb">set</span><span class="p">(</span><span class="n">X_train</span><span class="o">.</span><span class="n">columns</span><span class="p">)</span> <span class="o">-</span> <span class="nb">set</span><span class="p">(</span><span class="n">X_test</span><span class="o">.</span><span class="n">columns</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="s2">"Missing in val:"</span><span class="p">,</span> <span class="n">missing_val_cols</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="s2">"Missing in test:"</span><span class="p">,</span> <span class="n">missing_test_cols</span><span class="p">)</span>

<span class="nb">print</span><span class="p">(</span><span class="s2">"NaNs in train:"</span><span class="p">,</span> <span class="n">X_train</span><span class="o">.</span><span class="n">isna</span><span class="p">()</span><span class="o">.</span><span class="n">sum</span><span class="p">()</span><span class="o">.</span><span class="n">sum</span><span class="p">())</span>
<span class="nb">print</span><span class="p">(</span><span class="s2">"NaNs in val:"</span><span class="p">,</span> <span class="n">X_val</span><span class="o">.</span><span class="n">isna</span><span class="p">()</span><span class="o">.</span><span class="n">sum</span><span class="p">()</span><span class="o">.</span><span class="n">sum</span><span class="p">())</span>
<span class="nb">print</span><span class="p">(</span><span class="s2">"NaNs in test:"</span><span class="p">,</span> <span class="n">X_test</span><span class="o">.</span><span class="n">isna</span><span class="p">()</span><span class="o">.</span><span class="n">sum</span><span class="p">()</span><span class="o">.</span><span class="n">sum</span><span class="p">())</span>
</pre></div>
</div>
</div>
</div>
</div>
<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>
<div class="jp-OutputArea jp-Cell-outputArea">
<div class="jp-OutputArea-child">
<div class="jp-OutputPrompt jp-OutputArea-prompt"></div>
<div class="jp-RenderedText jp-OutputArea-output" data-mime-type="text/plain" tabindex="0">
<pre>Train shape: (243, 16)
Val shape:   (61, 16)
Test shape:  (77, 16)
Missing in val: set()
Missing in test: set()
NaNs in train: 0
NaNs in val: 0
NaNs in test: 0
</pre>
</div>
</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell" id="cell-id=caa1aae6-40de-493e-8496-ae04d3b8e80e">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In [20]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
<div class="cm-editor cm-s-jupyter">
<div class="highlight hl-ipython3"><pre><span></span><span class="c1"># Resample</span>
<span class="n">train_df</span> <span class="o">=</span> <span class="n">X_train</span><span class="o">.</span><span class="n">copy</span><span class="p">()</span>
<span class="n">train_df</span><span class="p">[</span><span class="s1">'target'</span><span class="p">]</span> <span class="o">=</span> <span class="n">y_train</span>
<span class="n">classes</span> <span class="o">=</span> <span class="n">train_df</span><span class="p">[</span><span class="s1">'target'</span><span class="p">]</span><span class="o">.</span><span class="n">unique</span><span class="p">()</span>
<span class="n">class_dfs</span> <span class="o">=</span> <span class="p">{</span><span class="n">c</span><span class="p">:</span> <span class="n">train_df</span><span class="p">[</span><span class="n">train_df</span><span class="p">[</span><span class="s1">'target'</span><span class="p">]</span> <span class="o">==</span> <span class="n">c</span><span class="p">]</span> <span class="k">for</span> <span class="n">c</span> <span class="ow">in</span> <span class="n">classes</span><span class="p">}</span>
<span class="n">max_size</span> <span class="o">=</span> <span class="n">train_df</span><span class="p">[</span><span class="s1">'target'</span><span class="p">]</span><span class="o">.</span><span class="n">value_counts</span><span class="p">()</span><span class="o">.</span><span class="n">max</span><span class="p">()</span>

<span class="n">resampled_dfs</span> <span class="o">=</span> <span class="p">[]</span>
<span class="k">for</span> <span class="n">c</span><span class="p">,</span> <span class="n">df</span> <span class="ow">in</span> <span class="n">class_dfs</span><span class="o">.</span><span class="n">items</span><span class="p">():</span>
    <span class="k">if</span> <span class="nb">len</span><span class="p">(</span><span class="n">df</span><span class="p">)</span> <span class="o">&lt;</span> <span class="n">max_size</span><span class="p">:</span>
        <span class="n">df_resampled</span> <span class="o">=</span> <span class="n">resample</span><span class="p">(</span><span class="n">df</span><span class="p">,</span> <span class="n">replace</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span> <span class="n">n_samples</span><span class="o">=</span><span class="n">max_size</span><span class="p">,</span> <span class="n">random_state</span><span class="o">=</span><span class="mi">42</span><span class="p">)</span>
    <span class="k">else</span><span class="p">:</span>
        <span class="n">df_resampled</span> <span class="o">=</span> <span class="n">df</span>
    <span class="n">resampled_dfs</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="n">df_resampled</span><span class="p">)</span>

<span class="n">train_balanced</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">concat</span><span class="p">(</span><span class="n">resampled_dfs</span><span class="p">)</span>
<span class="n">train_balanced</span> <span class="o">=</span> <span class="n">train_balanced</span><span class="o">.</span><span class="n">sample</span><span class="p">(</span><span class="n">frac</span><span class="o">=</span><span class="mi">1</span><span class="p">,</span> <span class="n">random_state</span><span class="o">=</span><span class="mi">42</span><span class="p">)</span><span class="o">.</span><span class="n">reset_index</span><span class="p">(</span><span class="n">drop</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>

<span class="n">X_train_bal</span> <span class="o">=</span> <span class="n">train_balanced</span><span class="o">.</span><span class="n">drop</span><span class="p">(</span><span class="s1">'target'</span><span class="p">,</span> <span class="n">axis</span><span class="o">=</span><span class="mi">1</span><span class="p">)</span>
<span class="n">y_train_bal</span> <span class="o">=</span> <span class="n">train_balanced</span><span class="p">[</span><span class="s1">'target'</span><span class="p">]</span>

<span class="nb">print</span><span class="p">(</span><span class="s2">"Before upsampling:"</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="n">pd</span><span class="o">.</span><span class="n">Series</span><span class="p">(</span><span class="n">y_train</span><span class="p">)</span><span class="o">.</span><span class="n">value_counts</span><span class="p">())</span>
<span class="nb">print</span><span class="p">(</span><span class="s2">"After upsampling:"</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="n">pd</span><span class="o">.</span><span class="n">Series</span><span class="p">(</span><span class="n">y_train_bal</span><span class="p">)</span><span class="o">.</span><span class="n">value_counts</span><span class="p">())</span>
<span class="nb">print</span><span class="p">(</span><span class="s2">"Number of features:"</span><span class="p">,</span> <span class="n">X_train_bal</span><span class="o">.</span><span class="n">shape</span><span class="p">[</span><span class="mi">1</span><span class="p">])</span>
<span class="nb">print</span><span class="p">(</span><span class="s2">"X_train_balanced shape:"</span><span class="p">,</span> <span class="n">y_train_bal</span><span class="o">.</span><span class="n">shape</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="s2">"y_train_balanced shape:"</span><span class="p">,</span> <span class="n">y_train_bal</span><span class="o">.</span><span class="n">shape</span><span class="p">)</span>
</pre></div>
</div>
</div>
</div>
</div>
<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>
<div class="jp-OutputArea jp-Cell-outputArea">
<div class="jp-OutputArea-child">
<div class="jp-OutputPrompt jp-OutputArea-prompt"></div>
<div class="jp-RenderedText jp-OutputArea-output" data-mime-type="text/plain" tabindex="0">
<pre>Before upsampling:
1    173
0     70
Name: count, dtype: int64
After upsampling:
target
1    173
0    173
Name: count, dtype: int64
Number of features: 16
X_train_balanced shape: (346,)
y_train_balanced shape: (346,)
</pre>
</div>
</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs" id="cell-id=746b3142-5266-4267-a2ee-9787e0cc7ca0">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In [21]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
<div class="cm-editor cm-s-jupyter">
<div class="highlight hl-ipython3"><pre><span></span><span class="c1"># Convert to tensors</span>
<span class="n">X_train_tensor</span> <span class="o">=</span> <span class="n">torch</span><span class="o">.</span><span class="n">tensor</span><span class="p">(</span><span class="n">X_train_bal</span><span class="o">.</span><span class="n">values</span><span class="p">,</span> <span class="n">dtype</span><span class="o">=</span><span class="n">torch</span><span class="o">.</span><span class="n">float32</span><span class="p">)</span>
<span class="n">y_train_tensor</span> <span class="o">=</span> <span class="n">torch</span><span class="o">.</span><span class="n">tensor</span><span class="p">(</span><span class="n">y_train_bal</span><span class="p">,</span> <span class="n">dtype</span><span class="o">=</span><span class="n">torch</span><span class="o">.</span><span class="n">long</span><span class="p">)</span>
<span class="n">X_val_tensor</span> <span class="o">=</span> <span class="n">torch</span><span class="o">.</span><span class="n">tensor</span><span class="p">(</span><span class="n">X_val</span><span class="o">.</span><span class="n">values</span><span class="p">,</span> <span class="n">dtype</span><span class="o">=</span><span class="n">torch</span><span class="o">.</span><span class="n">float32</span><span class="p">)</span>
<span class="n">y_val_tensor</span> <span class="o">=</span> <span class="n">torch</span><span class="o">.</span><span class="n">tensor</span><span class="p">(</span><span class="n">y_val</span><span class="p">,</span> <span class="n">dtype</span><span class="o">=</span><span class="n">torch</span><span class="o">.</span><span class="n">long</span><span class="p">)</span>
<span class="n">X_test_tensor</span> <span class="o">=</span> <span class="n">torch</span><span class="o">.</span><span class="n">tensor</span><span class="p">(</span><span class="n">X_test</span><span class="o">.</span><span class="n">values</span><span class="p">,</span> <span class="n">dtype</span><span class="o">=</span><span class="n">torch</span><span class="o">.</span><span class="n">float32</span><span class="p">)</span>
<span class="n">y_test_tensor</span>  <span class="o">=</span> <span class="n">torch</span><span class="o">.</span><span class="n">tensor</span><span class="p">(</span><span class="n">y_test</span><span class="p">,</span> <span class="n">dtype</span><span class="o">=</span><span class="n">torch</span><span class="o">.</span><span class="n">long</span><span class="p">)</span>
</pre></div>
</div>
</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell" id="cell-id=eff7a704-29e1-4006-b277-89ed436db0b6">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In [22]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
<div class="cm-editor cm-s-jupyter">
<div class="highlight hl-ipython3"><pre><span></span><span class="c1"># DataLoaders</span>
<span class="n">train_ds</span> <span class="o">=</span> <span class="n">TensorDataset</span><span class="p">(</span><span class="n">X_train_tensor</span><span class="p">,</span> <span class="n">y_train_tensor</span><span class="p">)</span>
<span class="n">val_ds</span> <span class="o">=</span> <span class="n">TensorDataset</span><span class="p">(</span><span class="n">X_val_tensor</span><span class="p">,</span> <span class="n">y_val_tensor</span><span class="p">)</span>
<span class="n">test_ds</span> <span class="o">=</span> <span class="n">TensorDataset</span><span class="p">(</span><span class="n">X_test_tensor</span><span class="p">,</span> <span class="n">y_test_tensor</span><span class="p">)</span>

<span class="n">train_loader</span> <span class="o">=</span> <span class="n">DataLoader</span><span class="p">(</span><span class="n">train_ds</span><span class="p">,</span> <span class="n">batch_size</span><span class="o">=</span><span class="mi">64</span><span class="p">,</span> <span class="n">shuffle</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>
<span class="n">val_loader</span> <span class="o">=</span> <span class="n">DataLoader</span><span class="p">(</span><span class="n">val_ds</span><span class="p">,</span> <span class="n">batch_size</span><span class="o">=</span><span class="mi">64</span><span class="p">)</span>
<span class="n">test_loader</span> <span class="o">=</span> <span class="n">DataLoader</span><span class="p">(</span><span class="n">test_ds</span><span class="p">,</span> <span class="n">batch_size</span><span class="o">=</span><span class="mi">64</span><span class="p">)</span>

<span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s2">"Train: </span><span class="si">{</span><span class="nb">len</span><span class="p">(</span><span class="n">train_ds</span><span class="p">)</span><span class="si">}</span><span class="s2">, Val: </span><span class="si">{</span><span class="nb">len</span><span class="p">(</span><span class="n">val_ds</span><span class="p">)</span><span class="si">}</span><span class="s2">, Test: </span><span class="si">{</span><span class="nb">len</span><span class="p">(</span><span class="n">test_ds</span><span class="p">)</span><span class="si">}</span><span class="s2">"</span><span class="p">)</span>
</pre></div>
</div>
</div>
</div>
</div>
<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>
<div class="jp-OutputArea jp-Cell-outputArea">
<div class="jp-OutputArea-child">
<div class="jp-OutputPrompt jp-OutputArea-prompt"></div>
<div class="jp-RenderedText jp-OutputArea-output" data-mime-type="text/plain" tabindex="0">
<pre>Train: 346, Val: 61, Test: 77
</pre>
</div>
</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell" id="cell-id=505d1520-2bbf-4c7f-bf60-f3b7415e1e96">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In [23]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
<div class="cm-editor cm-s-jupyter">
<div class="highlight hl-ipython3"><pre><span></span><span class="c1"># Model</span>
<span class="k">class</span><span class="w"> </span><span class="nc">NN</span><span class="p">(</span><span class="n">nn</span><span class="o">.</span><span class="n">Module</span><span class="p">):</span>
    <span class="k">def</span><span class="w"> </span><span class="fm">__init__</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">num_input</span><span class="p">):</span>
        <span class="nb">super</span><span class="p">()</span><span class="o">.</span><span class="fm">__init__</span><span class="p">()</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">fc1</span> <span class="o">=</span> <span class="n">nn</span><span class="o">.</span><span class="n">Linear</span><span class="p">(</span><span class="n">num_input</span><span class="p">,</span> <span class="mi">256</span><span class="p">)</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">bn1</span> <span class="o">=</span> <span class="n">nn</span><span class="o">.</span><span class="n">BatchNorm1d</span><span class="p">(</span><span class="mi">256</span><span class="p">)</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">fc2</span> <span class="o">=</span> <span class="n">nn</span><span class="o">.</span><span class="n">Linear</span><span class="p">(</span><span class="mi">256</span><span class="p">,</span> <span class="mi">128</span><span class="p">)</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">bn2</span> <span class="o">=</span> <span class="n">nn</span><span class="o">.</span><span class="n">BatchNorm1d</span><span class="p">(</span><span class="mi">128</span><span class="p">)</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">out</span> <span class="o">=</span> <span class="n">nn</span><span class="o">.</span><span class="n">Linear</span><span class="p">(</span><span class="mi">128</span><span class="p">,</span> <span class="mi">1</span><span class="p">)</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">act</span> <span class="o">=</span> <span class="n">nn</span><span class="o">.</span><span class="n">ReLU</span><span class="p">()</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">drop</span> <span class="o">=</span> <span class="n">nn</span><span class="o">.</span><span class="n">Dropout</span><span class="p">(</span><span class="mf">0.3</span><span class="p">)</span>

    <span class="k">def</span><span class="w"> </span><span class="nf">forward</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">x</span><span class="p">):</span>
        <span class="n">x</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">act</span><span class="p">(</span><span class="bp">self</span><span class="o">.</span><span class="n">bn1</span><span class="p">(</span><span class="bp">self</span><span class="o">.</span><span class="n">fc1</span><span class="p">(</span><span class="n">x</span><span class="p">)))</span>
        <span class="n">x</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">drop</span><span class="p">(</span><span class="n">x</span><span class="p">)</span>
        <span class="n">x</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">act</span><span class="p">(</span><span class="bp">self</span><span class="o">.</span><span class="n">bn2</span><span class="p">(</span><span class="bp">self</span><span class="o">.</span><span class="n">fc2</span><span class="p">(</span><span class="n">x</span><span class="p">)))</span>
        <span class="n">x</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">drop</span><span class="p">(</span><span class="n">x</span><span class="p">)</span>
        <span class="n">x</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">out</span><span class="p">(</span><span class="n">x</span><span class="p">)</span>         
        <span class="k">return</span> <span class="n">x</span><span class="o">.</span><span class="n">squeeze</span><span class="p">(</span><span class="mi">1</span><span class="p">)</span>

<span class="n">num_input</span> <span class="o">=</span> <span class="n">X_train_bal</span><span class="o">.</span><span class="n">shape</span><span class="p">[</span><span class="mi">1</span><span class="p">]</span>  
<span class="n">model</span> <span class="o">=</span> <span class="n">NN</span><span class="p">(</span><span class="n">num_input</span><span class="p">)</span><span class="o">.</span><span class="n">to</span><span class="p">(</span><span class="n">device</span><span class="p">)</span>

<span class="nb">print</span><span class="p">(</span><span class="n">model</span><span class="p">)</span>
<span class="nb">sum</span><span class="p">(</span><span class="n">p</span><span class="o">.</span><span class="n">numel</span><span class="p">()</span> <span class="k">for</span> <span class="n">p</span> <span class="ow">in</span> <span class="n">model</span><span class="o">.</span><span class="n">parameters</span><span class="p">())</span>
</pre></div>
</div>
</div>
</div>
</div>
<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>
<div class="jp-OutputArea jp-Cell-outputArea">
<div class="jp-OutputArea-child">
<div class="jp-OutputPrompt jp-OutputArea-prompt"></div>
<div class="jp-RenderedText jp-OutputArea-output" data-mime-type="text/plain" tabindex="0">
<pre>NN(
  (fc1): Linear(in_features=16, out_features=256, bias=True)
  (bn1): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (fc2): Linear(in_features=256, out_features=128, bias=True)
  (bn2): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (out): Linear(in_features=128, out_features=1, bias=True)
  (act): ReLU()
  (drop): Dropout(p=0.3, inplace=False)
)
</pre>
</div>
</div>
<div class="jp-OutputArea-child jp-OutputArea-executeResult">
<div class="jp-OutputPrompt jp-OutputArea-prompt">Out[23]:</div>
<div class="jp-RenderedText jp-OutputArea-output jp-OutputArea-executeResult" data-mime-type="text/plain" tabindex="0">
<pre>38145</pre>
</div>
</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell" id="cell-id=9af45080-9b08-4cdf-84bf-dc119eae9918">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In [24]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
<div class="cm-editor cm-s-jupyter">
<div class="highlight hl-ipython3"><pre><span></span><span class="n">loss_fn</span> <span class="o">=</span> <span class="n">nn</span><span class="o">.</span><span class="n">BCEWithLogitsLoss</span><span class="p">()</span> 
<span class="n">optimizer</span> <span class="o">=</span> <span class="n">optim</span><span class="o">.</span><span class="n">Adam</span><span class="p">(</span><span class="n">model</span><span class="o">.</span><span class="n">parameters</span><span class="p">(),</span> <span class="n">lr</span><span class="o">=</span><span class="n">lr</span><span class="p">)</span>
<span class="n">scheduler</span> <span class="o">=</span> <span class="n">optim</span><span class="o">.</span><span class="n">lr_scheduler</span><span class="o">.</span><span class="n">ReduceLROnPlateau</span><span class="p">(</span><span class="n">optimizer</span><span class="p">,</span> <span class="n">mode</span><span class="o">=</span><span class="s1">'max'</span><span class="p">,</span> <span class="n">patience</span><span class="o">=</span><span class="mi">5</span><span class="p">,</span> <span class="n">factor</span><span class="o">=</span><span class="mf">0.5</span><span class="p">)</span>

<span class="n">best_model_state</span> <span class="o">=</span> <span class="kc">None</span>
<span class="n">best_auc</span> <span class="o">=</span> <span class="mf">0.0</span>
<span class="n">patience_counter</span> <span class="o">=</span> <span class="mi">0</span>
<span class="n">patience</span> <span class="o">=</span> <span class="mi">17</span> 

<span class="k">for</span> <span class="n">epoch</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="n">num_epochs</span><span class="p">):</span>
    <span class="n">model</span><span class="o">.</span><span class="n">train</span><span class="p">()</span>
    <span class="n">running_loss</span> <span class="o">=</span> <span class="mf">0.0</span>
    <span class="k">for</span> <span class="n">Xb</span><span class="p">,</span> <span class="n">yb</span> <span class="ow">in</span> <span class="n">train_loader</span><span class="p">:</span>
        <span class="n">Xb</span> <span class="o">=</span> <span class="n">Xb</span><span class="o">.</span><span class="n">to</span><span class="p">(</span><span class="n">device</span><span class="p">);</span> <span class="n">yb</span> <span class="o">=</span> <span class="n">yb</span><span class="o">.</span><span class="n">to</span><span class="p">(</span><span class="n">device</span><span class="p">)</span><span class="o">.</span><span class="n">float</span><span class="p">()</span>
        <span class="n">optimizer</span><span class="o">.</span><span class="n">zero_grad</span><span class="p">()</span>
        <span class="n">logits</span> <span class="o">=</span> <span class="n">model</span><span class="p">(</span><span class="n">Xb</span><span class="p">)</span>
        <span class="n">loss</span> <span class="o">=</span> <span class="n">loss_fn</span><span class="p">(</span><span class="n">logits</span><span class="p">,</span> <span class="n">yb</span><span class="p">)</span>
        <span class="n">loss</span><span class="o">.</span><span class="n">backward</span><span class="p">()</span>
        <span class="n">optimizer</span><span class="o">.</span><span class="n">step</span><span class="p">()</span>
        <span class="n">running_loss</span> <span class="o">+=</span> <span class="n">loss</span><span class="o">.</span><span class="n">item</span><span class="p">()</span> <span class="o">*</span> <span class="n">Xb</span><span class="o">.</span><span class="n">size</span><span class="p">(</span><span class="mi">0</span><span class="p">)</span>

    <span class="n">train_loss</span> <span class="o">=</span> <span class="n">running_loss</span> <span class="o">/</span> <span class="nb">len</span><span class="p">(</span><span class="n">train_loader</span><span class="o">.</span><span class="n">dataset</span><span class="p">)</span>
    
    <span class="n">model</span><span class="o">.</span><span class="n">eval</span><span class="p">()</span>
    <span class="n">val_logits</span> <span class="o">=</span> <span class="p">[]</span>
    <span class="n">val_labels</span> <span class="o">=</span> <span class="p">[]</span>

    <span class="k">with</span> <span class="n">torch</span><span class="o">.</span><span class="n">no_grad</span><span class="p">():</span>
        <span class="k">for</span> <span class="n">Xb</span><span class="p">,</span> <span class="n">yb</span> <span class="ow">in</span> <span class="n">val_loader</span><span class="p">:</span>
            <span class="n">Xb</span> <span class="o">=</span> <span class="n">Xb</span><span class="o">.</span><span class="n">to</span><span class="p">(</span><span class="n">device</span><span class="p">)</span>
            <span class="n">logits</span> <span class="o">=</span> <span class="n">model</span><span class="p">(</span><span class="n">Xb</span><span class="p">)</span>
            <span class="n">val_logits</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="n">logits</span><span class="o">.</span><span class="n">cpu</span><span class="p">())</span>
            <span class="n">val_labels</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="n">yb</span><span class="p">)</span>

    <span class="n">val_logits</span> <span class="o">=</span> <span class="n">torch</span><span class="o">.</span><span class="n">cat</span><span class="p">(</span><span class="n">val_logits</span><span class="p">)</span>
    <span class="n">val_labels</span> <span class="o">=</span> <span class="n">torch</span><span class="o">.</span><span class="n">cat</span><span class="p">(</span><span class="n">val_labels</span><span class="p">)</span>
    <span class="n">val_probs</span> <span class="o">=</span> <span class="n">torch</span><span class="o">.</span><span class="n">sigmoid</span><span class="p">(</span><span class="n">val_logits</span><span class="p">)</span><span class="o">.</span><span class="n">numpy</span><span class="p">()</span>
    <span class="n">val_auc</span> <span class="o">=</span> <span class="n">roc_auc_score</span><span class="p">(</span><span class="n">val_labels</span><span class="o">.</span><span class="n">numpy</span><span class="p">(),</span> <span class="n">val_probs</span><span class="p">)</span>

    <span class="k">if</span> <span class="n">val_auc</span> <span class="o">&gt;</span> <span class="n">best_auc</span><span class="p">:</span>
        <span class="n">best_auc</span> <span class="o">=</span> <span class="n">val_auc</span>
        <span class="n">best_model_state</span> <span class="o">=</span> <span class="n">model</span><span class="o">.</span><span class="n">state_dict</span><span class="p">()</span><span class="o">.</span><span class="n">copy</span><span class="p">()</span>
        <span class="n">patience_counter</span> <span class="o">=</span> <span class="mi">0</span>
    <span class="k">else</span><span class="p">:</span>
        <span class="n">patience_counter</span> <span class="o">+=</span> <span class="mi">1</span>
        <span class="k">if</span> <span class="n">patience_counter</span> <span class="o">&gt;=</span> <span class="n">patience</span><span class="p">:</span>
            <span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s2">"Early stopping at epoch </span><span class="si">{</span><span class="n">epoch</span><span class="o">+</span><span class="mi">1</span><span class="si">}</span><span class="s2">"</span><span class="p">)</span>
            <span class="k">break</span>

    <span class="n">scheduler</span><span class="o">.</span><span class="n">step</span><span class="p">(</span><span class="n">val_auc</span><span class="p">)</span>

    <span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s2">"Epoch </span><span class="si">{</span><span class="n">epoch</span><span class="o">+</span><span class="mi">1</span><span class="si">}</span><span class="s2">/</span><span class="si">{</span><span class="n">num_epochs</span><span class="si">}</span><span class="s2"> | Train loss: </span><span class="si">{</span><span class="n">train_loss</span><span class="si">:</span><span class="s2">.6f</span><span class="si">}</span><span class="s2"> | Val AUC: </span><span class="si">{</span><span class="n">val_auc</span><span class="si">:</span><span class="s2">.4f</span><span class="si">}</span><span class="s2">"</span><span class="p">)</span>

<span class="n">model</span><span class="o">.</span><span class="n">load_state_dict</span><span class="p">(</span><span class="n">best_model_state</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s2">"Best model (val_auc=</span><span class="si">{</span><span class="n">best_auc</span><span class="si">:</span><span class="s2">.4f</span><span class="si">}</span><span class="s2">) restored"</span><span class="p">)</span>
</pre></div>
</div>
</div>
</div>
</div>
<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>
<div class="jp-OutputArea jp-Cell-outputArea">
<div class="jp-OutputArea-child">
<div class="jp-OutputPrompt jp-OutputArea-prompt"></div>
<div class="jp-RenderedText jp-OutputArea-output" data-mime-type="text/plain" tabindex="0">
<pre>Epoch 1/75 | Train loss: 0.694142 | Val AUC: 0.6550
Epoch 2/75 | Train loss: 0.670326 | Val AUC: 0.7041
Epoch 3/75 | Train loss: 0.657135 | Val AUC: 0.7054
Epoch 4/75 | Train loss: 0.655149 | Val AUC: 0.6292
Epoch 5/75 | Train loss: 0.623411 | Val AUC: 0.6357
Epoch 6/75 | Train loss: 0.612246 | Val AUC: 0.6718
Epoch 7/75 | Train loss: 0.585797 | Val AUC: 0.6809
Epoch 8/75 | Train loss: 0.543807 | Val AUC: 0.6292
Epoch 9/75 | Train loss: 0.563655 | Val AUC: 0.6822
Epoch 10/75 | Train loss: 0.534841 | Val AUC: 0.6938
Epoch 11/75 | Train loss: 0.508537 | Val AUC: 0.6886
Epoch 12/75 | Train loss: 0.508282 | Val AUC: 0.6912
Epoch 13/75 | Train loss: 0.520399 | Val AUC: 0.6951
Epoch 14/75 | Train loss: 0.474620 | Val AUC: 0.7106
Epoch 15/75 | Train loss: 0.477129 | Val AUC: 0.7313
Epoch 16/75 | Train loss: 0.465939 | Val AUC: 0.7442
Epoch 17/75 | Train loss: 0.437644 | Val AUC: 0.7442
Epoch 18/75 | Train loss: 0.439457 | Val AUC: 0.7222
Epoch 19/75 | Train loss: 0.431787 | Val AUC: 0.7377
Epoch 20/75 | Train loss: 0.428295 | Val AUC: 0.7829
Epoch 21/75 | Train loss: 0.434153 | Val AUC: 0.7829
Epoch 22/75 | Train loss: 0.406786 | Val AUC: 0.7933
Epoch 23/75 | Train loss: 0.426460 | Val AUC: 0.7894
Epoch 24/75 | Train loss: 0.405213 | Val AUC: 0.7649
Epoch 25/75 | Train loss: 0.396205 | Val AUC: 0.7984
Epoch 26/75 | Train loss: 0.399268 | Val AUC: 0.8062
Epoch 27/75 | Train loss: 0.383469 | Val AUC: 0.7765
Epoch 28/75 | Train loss: 0.377538 | Val AUC: 0.8101
Epoch 29/75 | Train loss: 0.352117 | Val AUC: 0.8101
Epoch 30/75 | Train loss: 0.378090 | Val AUC: 0.8049
Epoch 31/75 | Train loss: 0.373611 | Val AUC: 0.7855
Epoch 32/75 | Train loss: 0.377552 | Val AUC: 0.8062
Epoch 33/75 | Train loss: 0.352395 | Val AUC: 0.7972
Epoch 34/75 | Train loss: 0.374261 | Val AUC: 0.7817
Epoch 35/75 | Train loss: 0.339617 | Val AUC: 0.7804
Epoch 36/75 | Train loss: 0.358721 | Val AUC: 0.7791
Epoch 37/75 | Train loss: 0.343269 | Val AUC: 0.7778
Epoch 38/75 | Train loss: 0.365080 | Val AUC: 0.7855
Epoch 39/75 | Train loss: 0.328033 | Val AUC: 0.7907
Epoch 40/75 | Train loss: 0.350436 | Val AUC: 0.7829
Epoch 41/75 | Train loss: 0.352447 | Val AUC: 0.7791
Epoch 42/75 | Train loss: 0.324213 | Val AUC: 0.7829
Epoch 43/75 | Train loss: 0.340579 | Val AUC: 0.7881
Epoch 44/75 | Train loss: 0.319216 | Val AUC: 0.7907
Early stopping at epoch 45
Best model (val_auc=0.8101) restored
</pre>
</div>
</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell" id="cell-id=6c656f97-a27f-42d9-a997-e5edd56201ce">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In [25]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
<div class="cm-editor cm-s-jupyter">
<div class="highlight hl-ipython3"><pre><span></span><span class="n">model</span><span class="o">.</span><span class="n">eval</span><span class="p">()</span>
<span class="n">y_val_probs</span> <span class="o">=</span> <span class="p">[]</span>
<span class="k">with</span> <span class="n">torch</span><span class="o">.</span><span class="n">no_grad</span><span class="p">():</span>
    <span class="k">for</span> <span class="n">X_batch</span><span class="p">,</span> <span class="n">_</span> <span class="ow">in</span> <span class="n">val_loader</span><span class="p">:</span>
        <span class="n">X_batch</span> <span class="o">=</span> <span class="n">X_batch</span><span class="o">.</span><span class="n">to</span><span class="p">(</span><span class="n">device</span><span class="p">)</span>
        <span class="n">outputs</span> <span class="o">=</span> <span class="n">model</span><span class="p">(</span><span class="n">X_batch</span><span class="p">)</span>
        <span class="n">probs</span> <span class="o">=</span> <span class="n">torch</span><span class="o">.</span><span class="n">sigmoid</span><span class="p">(</span><span class="n">outputs</span><span class="p">)</span>  
        <span class="n">y_val_probs</span><span class="o">.</span><span class="n">extend</span><span class="p">(</span><span class="n">probs</span><span class="o">.</span><span class="n">cpu</span><span class="p">()</span><span class="o">.</span><span class="n">numpy</span><span class="p">())</span>
<span class="n">y_val_probs</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">(</span><span class="n">y_val_probs</span><span class="p">)</span>
<span class="n">y_val_true</span> <span class="o">=</span> <span class="n">y_val</span> 

<span class="n">prec</span><span class="p">,</span> <span class="n">rec</span><span class="p">,</span> <span class="n">thresholds</span> <span class="o">=</span> <span class="n">precision_recall_curve</span><span class="p">(</span><span class="n">y_val_true</span><span class="p">,</span> <span class="n">y_val_probs</span><span class="p">)</span>
<span class="n">denom</span> <span class="o">=</span> <span class="n">prec</span> <span class="o">+</span> <span class="n">rec</span>
<span class="n">f1</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">zeros_like</span><span class="p">(</span><span class="n">denom</span><span class="p">)</span>
<span class="n">mask</span> <span class="o">=</span> <span class="n">denom</span> <span class="o">!=</span> <span class="mi">0</span>
<span class="n">f1</span><span class="p">[</span><span class="n">mask</span><span class="p">]</span> <span class="o">=</span> <span class="mi">2</span> <span class="o">*</span> <span class="n">prec</span><span class="p">[</span><span class="n">mask</span><span class="p">]</span> <span class="o">*</span> <span class="n">rec</span><span class="p">[</span><span class="n">mask</span><span class="p">]</span> <span class="o">/</span> <span class="n">denom</span><span class="p">[</span><span class="n">mask</span><span class="p">]</span>
<span class="n">best_thresh</span> <span class="o">=</span> <span class="n">thresholds</span><span class="p">[</span><span class="n">np</span><span class="o">.</span><span class="n">argmax</span><span class="p">(</span><span class="n">f1</span><span class="p">[:</span><span class="o">-</span><span class="mi">1</span><span class="p">])]</span>
<span class="nb">print</span><span class="p">(</span><span class="s2">"Best threshold from validation set for F1:"</span><span class="p">,</span> <span class="n">best_thresh</span><span class="p">)</span>

<span class="n">y_test_probs</span> <span class="o">=</span> <span class="p">[]</span>
<span class="k">with</span> <span class="n">torch</span><span class="o">.</span><span class="n">no_grad</span><span class="p">():</span>
    <span class="k">for</span> <span class="n">X_batch</span><span class="p">,</span> <span class="n">_</span> <span class="ow">in</span> <span class="n">test_loader</span><span class="p">:</span>
        <span class="n">X_batch</span> <span class="o">=</span> <span class="n">X_batch</span><span class="o">.</span><span class="n">to</span><span class="p">(</span><span class="n">device</span><span class="p">)</span>
        <span class="n">outputs</span> <span class="o">=</span> <span class="n">model</span><span class="p">(</span><span class="n">X_batch</span><span class="p">)</span>
        <span class="n">probs</span> <span class="o">=</span> <span class="n">torch</span><span class="o">.</span><span class="n">sigmoid</span><span class="p">(</span><span class="n">outputs</span><span class="p">)</span>
        <span class="n">y_test_probs</span><span class="o">.</span><span class="n">extend</span><span class="p">(</span><span class="n">probs</span><span class="o">.</span><span class="n">cpu</span><span class="p">()</span><span class="o">.</span><span class="n">numpy</span><span class="p">())</span>
<span class="n">y_test_probs</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">(</span><span class="n">y_test_probs</span><span class="p">)</span>
<span class="n">y_test_pred_opt</span> <span class="o">=</span> <span class="p">(</span><span class="n">y_test_probs</span> <span class="o">&gt;</span> <span class="n">best_thresh</span><span class="p">)</span><span class="o">.</span><span class="n">astype</span><span class="p">(</span><span class="nb">int</span><span class="p">)</span>

<span class="n">target_names</span> <span class="o">=</span> <span class="p">[</span><span class="s1">'Repaid'</span><span class="p">,</span> <span class="s1">'Defaulted'</span><span class="p">]</span>
<span class="n">report</span> <span class="o">=</span> <span class="n">classification_report</span><span class="p">(</span><span class="n">y_test</span><span class="p">,</span> <span class="n">y_test_pred_opt</span><span class="p">,</span> <span class="n">target_names</span><span class="o">=</span><span class="n">target_names</span><span class="p">)</span>
<span class="n">acc</span> <span class="o">=</span> <span class="n">accuracy_score</span><span class="p">(</span><span class="n">y_test</span><span class="p">,</span> <span class="n">y_test_pred_opt</span><span class="p">)</span>
<span class="n">roc_auc</span> <span class="o">=</span> <span class="n">roc_auc_score</span><span class="p">(</span><span class="n">y_test</span><span class="p">,</span> <span class="n">y_test_probs</span><span class="p">)</span>
<span class="n">cm</span> <span class="o">=</span> <span class="n">confusion_matrix</span><span class="p">(</span><span class="n">y_test</span><span class="p">,</span> <span class="n">y_test_pred_opt</span><span class="p">)</span>
<span class="n">tn</span><span class="p">,</span> <span class="n">fp</span><span class="p">,</span> <span class="n">fn</span><span class="p">,</span> <span class="n">tp</span> <span class="o">=</span> <span class="n">cm</span><span class="o">.</span><span class="n">ravel</span><span class="p">()</span>
<span class="n">per_class_acc</span> <span class="o">=</span> <span class="n">cm</span><span class="o">.</span><span class="n">diagonal</span><span class="p">()</span> <span class="o">/</span> <span class="n">cm</span><span class="o">.</span><span class="n">sum</span><span class="p">(</span><span class="n">axis</span><span class="o">=</span><span class="mi">1</span><span class="p">)</span>

<span class="nb">print</span><span class="p">(</span><span class="n">report</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s2">"Accuracy: </span><span class="si">{</span><span class="n">acc</span><span class="o">*</span><span class="mi">100</span><span class="si">:</span><span class="s2">.2f</span><span class="si">}</span><span class="s2">%"</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s2">"ROC AUC: </span><span class="si">{</span><span class="n">roc_auc</span><span class="si">:</span><span class="s2">.3f</span><span class="si">}</span><span class="s2">"</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s2">"TP=</span><span class="si">{</span><span class="n">tp</span><span class="si">}</span><span class="s2">, FP=</span><span class="si">{</span><span class="n">fp</span><span class="si">}</span><span class="s2">, TN=</span><span class="si">{</span><span class="n">tn</span><span class="si">}</span><span class="s2">, FN=</span><span class="si">{</span><span class="n">fn</span><span class="si">}</span><span class="s2">"</span><span class="p">)</span>
<span class="k">for</span> <span class="n">i</span><span class="p">,</span> <span class="n">class_name</span> <span class="ow">in</span> <span class="nb">enumerate</span><span class="p">(</span><span class="n">target_names</span><span class="p">):</span>
    <span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s2">"Accuracy for class '</span><span class="si">{</span><span class="n">class_name</span><span class="si">}</span><span class="s2">': </span><span class="si">{</span><span class="n">per_class_acc</span><span class="p">[</span><span class="n">i</span><span class="p">]</span><span class="o">*</span><span class="mi">100</span><span class="si">:</span><span class="s2">.2f</span><span class="si">}</span><span class="s2">%"</span><span class="p">)</span>

<span class="n">plt</span><span class="o">.</span><span class="n">figure</span><span class="p">(</span><span class="n">figsize</span><span class="o">=</span><span class="p">(</span><span class="mi">6</span><span class="p">,</span><span class="mi">5</span><span class="p">))</span>
<span class="n">sns</span><span class="o">.</span><span class="n">heatmap</span><span class="p">(</span><span class="n">cm</span><span class="p">,</span> <span class="n">annot</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span> <span class="n">fmt</span><span class="o">=</span><span class="s1">'d'</span><span class="p">,</span> <span class="n">cmap</span><span class="o">=</span><span class="s1">'Blues'</span><span class="p">,</span>
            <span class="n">xticklabels</span><span class="o">=</span><span class="n">target_names</span><span class="p">,</span> <span class="n">yticklabels</span><span class="o">=</span><span class="n">target_names</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">xlabel</span><span class="p">(</span><span class="s2">"Predicted"</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">ylabel</span><span class="p">(</span><span class="s2">"Actual"</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">title</span><span class="p">(</span><span class="sa">f</span><span class="s2">"Confusion Matrix (Threshold = </span><span class="si">{</span><span class="n">best_thresh</span><span class="si">:</span><span class="s2">.2f</span><span class="si">}</span><span class="s2">)"</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">show</span><span class="p">()</span>
</pre></div>
</div>
</div>
</div>
</div>
<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>
<div class="jp-OutputArea jp-Cell-outputArea">
<div class="jp-OutputArea-child">
<div class="jp-OutputPrompt jp-OutputArea-prompt"></div>
<div class="jp-RenderedText jp-OutputArea-output" data-mime-type="text/plain" tabindex="0">
<pre>Best threshold from validation set for F1: 0.29837355
              precision    recall  f1-score   support

      Repaid       0.86      0.55      0.67        22
   Defaulted       0.84      0.96      0.90        55

    accuracy                           0.84        77
   macro avg       0.85      0.75      0.78        77
weighted avg       0.85      0.84      0.83        77

Accuracy: 84.42%
ROC AUC: 0.818
TP=53, FP=10, TN=12, FN=2
Accuracy for class 'Repaid': 54.55%
Accuracy for class 'Defaulted': 96.36%
</pre>
</div>
</div>
<div class="jp-OutputArea-child">
<div class="jp-OutputPrompt jp-OutputArea-prompt"></div>
<div class="jp-RenderedImage jp-OutputArea-output" tabindex="0">
<img alt="No description has been provided for this image" class="" src="data:image/png;base64,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"/>
</div>
</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs" id="cell-id=c814fb59-cf6d-4ae8-89f4-70c85f644016">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In [26]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
<div class="cm-editor cm-s-jupyter">
<div class="highlight hl-ipython3"><pre><span></span><span class="c1"># Data sets</span>
<span class="n">dtrain</span> <span class="o">=</span> <span class="n">xgb</span><span class="o">.</span><span class="n">DMatrix</span><span class="p">(</span><span class="n">X_train_bal</span><span class="p">,</span> <span class="n">label</span><span class="o">=</span><span class="n">y_train_bal</span><span class="p">)</span>
<span class="n">dval</span> <span class="o">=</span> <span class="n">xgb</span><span class="o">.</span><span class="n">DMatrix</span><span class="p">(</span><span class="n">X_val</span><span class="p">,</span> <span class="n">label</span><span class="o">=</span><span class="n">y_val</span><span class="p">)</span>
<span class="n">dtest</span> <span class="o">=</span> <span class="n">xgb</span><span class="o">.</span><span class="n">DMatrix</span><span class="p">(</span><span class="n">X_test</span><span class="p">,</span> <span class="n">label</span><span class="o">=</span><span class="n">y_test</span><span class="p">)</span> 
</pre></div>
</div>
</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs" id="cell-id=433db274-f23a-4922-9ecd-f5c41e0433fd">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In [27]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
<div class="cm-editor cm-s-jupyter">
<div class="highlight hl-ipython3"><pre><span></span><span class="c1"># Model</span>
<span class="n">params</span> <span class="o">=</span> <span class="p">{</span>
    <span class="s2">"objective"</span><span class="p">:</span> <span class="s2">"binary:logistic"</span><span class="p">,</span>
    <span class="s2">"eval_metric"</span><span class="p">:</span> <span class="s2">"logloss"</span><span class="p">,</span>
    <span class="s2">"eta"</span><span class="p">:</span> <span class="mf">0.05</span><span class="p">,</span>
    <span class="s2">"max_depth"</span><span class="p">:</span> <span class="mi">5</span><span class="p">,</span>
    <span class="s2">"subsample"</span><span class="p">:</span> <span class="mf">0.8</span><span class="p">,</span>
    <span class="s2">"colsample_bytree"</span><span class="p">:</span> <span class="mf">0.8</span><span class="p">,</span>
    <span class="s2">"lambda"</span><span class="p">:</span> <span class="mf">1.0</span><span class="p">,</span>
    <span class="s2">"seed"</span><span class="p">:</span> <span class="mi">42</span><span class="p">,</span>
<span class="p">}</span>

<span class="n">evals</span> <span class="o">=</span> <span class="p">[(</span><span class="n">dtrain</span><span class="p">,</span> <span class="s2">"train"</span><span class="p">),</span> <span class="p">(</span><span class="n">dval</span><span class="p">,</span> <span class="s2">"validation"</span><span class="p">)]</span>
</pre></div>
</div>
</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell" id="cell-id=f2953291-5c20-4a1f-851f-bf0453fa7a11">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In [28]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
<div class="cm-editor cm-s-jupyter">
<div class="highlight hl-ipython3"><pre><span></span><span class="c1"># Train</span>
<span class="n">model_b</span> <span class="o">=</span> <span class="n">xgb</span><span class="o">.</span><span class="n">train</span><span class="p">(</span>
    <span class="n">params</span><span class="o">=</span><span class="n">params</span><span class="p">,</span>
    <span class="n">dtrain</span><span class="o">=</span><span class="n">dtrain</span><span class="p">,</span>
    <span class="n">num_boost_round</span><span class="o">=</span><span class="mi">500</span><span class="p">,</span>
    <span class="n">evals</span><span class="o">=</span><span class="n">evals</span><span class="p">,</span>
    <span class="n">early_stopping_rounds</span><span class="o">=</span><span class="mi">50</span><span class="p">,</span>
    <span class="n">verbose_eval</span><span class="o">=</span><span class="mi">50</span>
<span class="p">)</span>
</pre></div>
</div>
</div>
</div>
</div>
<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>
<div class="jp-OutputArea jp-Cell-outputArea">
<div class="jp-OutputArea-child">
<div class="jp-OutputPrompt jp-OutputArea-prompt"></div>
<div class="jp-RenderedText jp-OutputArea-output" data-mime-type="text/plain" tabindex="0">
<pre>[0]	train-logloss:0.66429	validation-logloss:0.67654
[50]	train-logloss:0.23502	validation-logloss:0.47746
[100]	train-logloss:0.14951	validation-logloss:0.50338
[101]	train-logloss:0.14880	validation-logloss:0.50472
</pre>
</div>
</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell" id="cell-id=98fe75e1-344d-4237-b4ad-cf0bd41a6ea2">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In [29]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
<div class="cm-editor cm-s-jupyter">
<div class="highlight hl-ipython3"><pre><span></span><span class="c1"># Evaluation</span>
<span class="n">y_probs</span> <span class="o">=</span> <span class="n">model_b</span><span class="o">.</span><span class="n">predict</span><span class="p">(</span><span class="n">dtest</span><span class="p">)</span> 

<span class="n">prec</span><span class="p">,</span> <span class="n">rec</span><span class="p">,</span> <span class="n">thresholds</span> <span class="o">=</span> <span class="n">precision_recall_curve</span><span class="p">(</span><span class="n">y_test</span><span class="p">,</span> <span class="n">y_probs</span><span class="p">)</span> 
<span class="n">denom</span> <span class="o">=</span> <span class="n">prec</span> <span class="o">+</span> <span class="n">rec</span>
<span class="n">f1</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">zeros_like</span><span class="p">(</span><span class="n">denom</span><span class="p">)</span>
<span class="n">mask</span> <span class="o">=</span> <span class="n">denom</span> <span class="o">!=</span> <span class="mi">0</span>
<span class="n">f1</span><span class="p">[</span><span class="n">mask</span><span class="p">]</span> <span class="o">=</span> <span class="mi">2</span> <span class="o">*</span> <span class="n">prec</span><span class="p">[</span><span class="n">mask</span><span class="p">]</span> <span class="o">*</span> <span class="n">rec</span><span class="p">[</span><span class="n">mask</span><span class="p">]</span> <span class="o">/</span> <span class="n">denom</span><span class="p">[</span><span class="n">mask</span><span class="p">]</span>
<span class="n">best_thresh</span> <span class="o">=</span> <span class="n">thresholds</span><span class="p">[</span><span class="n">np</span><span class="o">.</span><span class="n">argmax</span><span class="p">(</span><span class="n">f1</span><span class="p">[:</span><span class="o">-</span><span class="mi">1</span><span class="p">])]</span>
<span class="n">y_pred_opt</span> <span class="o">=</span> <span class="p">(</span><span class="n">y_probs</span> <span class="o">&gt;</span> <span class="n">best_thresh</span><span class="p">)</span><span class="o">.</span><span class="n">astype</span><span class="p">(</span><span class="nb">int</span><span class="p">)</span>

<span class="n">target_names</span> <span class="o">=</span> <span class="p">[</span><span class="s1">'Repaid'</span><span class="p">,</span> <span class="s1">'Defaulted'</span><span class="p">]</span>
<span class="n">report</span> <span class="o">=</span> <span class="n">classification_report</span><span class="p">(</span><span class="n">y_test</span><span class="p">,</span> <span class="n">y_pred_opt</span><span class="p">,</span> <span class="n">target_names</span><span class="o">=</span><span class="n">target_names</span><span class="p">)</span>
<span class="n">acc</span> <span class="o">=</span> <span class="n">accuracy_score</span><span class="p">(</span><span class="n">y_test</span><span class="p">,</span> <span class="n">y_pred_opt</span><span class="p">)</span>
<span class="n">cm</span> <span class="o">=</span> <span class="n">confusion_matrix</span><span class="p">(</span><span class="n">y_test</span><span class="p">,</span> <span class="n">y_pred_opt</span><span class="p">)</span>
<span class="n">tn</span><span class="p">,</span> <span class="n">fp</span><span class="p">,</span> <span class="n">fn</span><span class="p">,</span> <span class="n">tp</span> <span class="o">=</span> <span class="n">cm</span><span class="o">.</span><span class="n">ravel</span><span class="p">()</span>
<span class="n">per_class_acc</span> <span class="o">=</span> <span class="n">cm</span><span class="o">.</span><span class="n">diagonal</span><span class="p">()</span> <span class="o">/</span> <span class="n">cm</span><span class="o">.</span><span class="n">sum</span><span class="p">(</span><span class="n">axis</span><span class="o">=</span><span class="mi">1</span><span class="p">)</span>
<span class="n">roc_auc</span> <span class="o">=</span> <span class="n">roc_auc_score</span><span class="p">(</span><span class="n">y_test</span><span class="p">,</span> <span class="n">y_probs</span><span class="p">)</span>

<span class="nb">print</span><span class="p">(</span><span class="s2">"Best threshold for F1:"</span><span class="p">,</span> <span class="n">best_thresh</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="n">report</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s2">"Accuracy: </span><span class="si">{</span><span class="n">acc</span><span class="o">*</span><span class="mi">100</span><span class="si">:</span><span class="s2">.2f</span><span class="si">}</span><span class="s2">%"</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s2">"ROC AUC: </span><span class="si">{</span><span class="n">roc_auc</span><span class="si">:</span><span class="s2">.3f</span><span class="si">}</span><span class="s2">"</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s2">"TP=</span><span class="si">{</span><span class="n">tp</span><span class="si">}</span><span class="s2">, FP=</span><span class="si">{</span><span class="n">fp</span><span class="si">}</span><span class="s2">, TN=</span><span class="si">{</span><span class="n">tn</span><span class="si">}</span><span class="s2">, FN=</span><span class="si">{</span><span class="n">fn</span><span class="si">}</span><span class="s2">"</span><span class="p">)</span>
<span class="k">for</span> <span class="n">i</span><span class="p">,</span> <span class="n">class_name</span> <span class="ow">in</span> <span class="nb">enumerate</span><span class="p">(</span><span class="n">target_names</span><span class="p">):</span>
    <span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s2">"Accuracy for class '</span><span class="si">{</span><span class="n">class_name</span><span class="si">}</span><span class="s2">': </span><span class="si">{</span><span class="n">per_class_acc</span><span class="p">[</span><span class="n">i</span><span class="p">]</span><span class="o">*</span><span class="mi">100</span><span class="si">:</span><span class="s2">.2f</span><span class="si">}</span><span class="s2">%"</span><span class="p">)</span>

<span class="n">plt</span><span class="o">.</span><span class="n">figure</span><span class="p">(</span><span class="n">figsize</span><span class="o">=</span><span class="p">(</span><span class="mi">6</span><span class="p">,</span><span class="mi">5</span><span class="p">))</span>
<span class="n">sns</span><span class="o">.</span><span class="n">heatmap</span><span class="p">(</span><span class="n">cm</span><span class="p">,</span> <span class="n">annot</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span> <span class="n">fmt</span><span class="o">=</span><span class="s1">'d'</span><span class="p">,</span> <span class="n">cmap</span><span class="o">=</span><span class="s1">'Blues'</span><span class="p">,</span>
            <span class="n">xticklabels</span><span class="o">=</span><span class="n">target_names</span><span class="p">,</span> <span class="n">yticklabels</span><span class="o">=</span><span class="n">target_names</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">xlabel</span><span class="p">(</span><span class="s2">"Predicted"</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">ylabel</span><span class="p">(</span><span class="s2">"Actual"</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">title</span><span class="p">(</span><span class="sa">f</span><span class="s2">"Confusion Matrix (Threshold = </span><span class="si">{</span><span class="n">best_thresh</span><span class="si">:</span><span class="s2">.2f</span><span class="si">}</span><span class="s2">)"</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">show</span><span class="p">()</span>

<span class="n">fpr</span><span class="p">,</span> <span class="n">tpr</span><span class="p">,</span> <span class="n">_</span> <span class="o">=</span> <span class="n">roc_curve</span><span class="p">(</span><span class="n">y_test</span><span class="p">,</span> <span class="n">y_probs</span><span class="p">)</span> 
<span class="n">plt</span><span class="o">.</span><span class="n">figure</span><span class="p">(</span><span class="n">figsize</span><span class="o">=</span><span class="p">(</span><span class="mi">6</span><span class="p">,</span><span class="mi">5</span><span class="p">))</span>
<span class="n">plt</span><span class="o">.</span><span class="n">plot</span><span class="p">(</span><span class="n">fpr</span><span class="p">,</span> <span class="n">tpr</span><span class="p">,</span> <span class="n">label</span><span class="o">=</span><span class="sa">f</span><span class="s2">"ROC curve (AUC = </span><span class="si">{</span><span class="n">roc_auc</span><span class="si">:</span><span class="s2">.3f</span><span class="si">}</span><span class="s2">)"</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">plot</span><span class="p">([</span><span class="mi">0</span><span class="p">,</span><span class="mi">1</span><span class="p">],</span> <span class="p">[</span><span class="mi">0</span><span class="p">,</span><span class="mi">1</span><span class="p">],</span> <span class="s1">'k--'</span><span class="p">,</span> <span class="n">label</span><span class="o">=</span><span class="s2">"Random"</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">xlabel</span><span class="p">(</span><span class="s2">"False Positive Rate"</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">ylabel</span><span class="p">(</span><span class="s2">"True Positive Rate"</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">title</span><span class="p">(</span><span class="s2">"ROC Curve"</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">legend</span><span class="p">()</span>
<span class="n">plt</span><span class="o">.</span><span class="n">show</span><span class="p">()</span>
</pre></div>
</div>
</div>
</div>
</div>
<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>
<div class="jp-OutputArea jp-Cell-outputArea">
<div class="jp-OutputArea-child">
<div class="jp-OutputPrompt jp-OutputArea-prompt"></div>
<div class="jp-RenderedText jp-OutputArea-output" data-mime-type="text/plain" tabindex="0">
<pre>Best threshold for F1: 0.39510784
              precision    recall  f1-score   support

      Repaid       0.81      0.59      0.68        22
   Defaulted       0.85      0.95      0.90        55

    accuracy                           0.84        77
   macro avg       0.83      0.77      0.79        77
weighted avg       0.84      0.84      0.84        77

Accuracy: 84.42%
ROC AUC: 0.827
TP=52, FP=9, TN=13, FN=3
Accuracy for class 'Repaid': 59.09%
Accuracy for class 'Defaulted': 94.55%
</pre>
</div>
</div>
<div class="jp-OutputArea-child">
<div class="jp-OutputPrompt jp-OutputArea-prompt"></div>
<div class="jp-RenderedImage jp-OutputArea-output" tabindex="0">
<img alt="No description has been provided for this image" class="" src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAfkAAAHWCAYAAAB0TPAHAAAAOnRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjEwLjUsIGh0dHBzOi8vbWF0cGxvdGxpYi5vcmcvWftoOwAAAAlwSFlzAAAPYQAAD2EBqD+naQAATAtJREFUeJzt3XdYFNf7NvB7EFh6U6QoIIoiWGOJEhIRJdYgRjRR41ewx2DDEiWmWKIYjRqjsSWxxhJ7NJZExRK7YjcEGwYLYAWkLQjn/cPX/bkBdNGFXWbvT665Ljkze+aZDcuz55kzM5IQQoCIiIhkx0jXARAREVHpYJInIiKSKSZ5IiIimWKSJyIikikmeSIiIplikiciIpIpJnkiIiKZYpInIiKSKSZ5IiIimWKSNyBXrlxBmzZtYGtrC0mSsGXLFq32f+PGDUiShGXLlmm13/KsZcuWaNmypVb7vHnzJszMzHD48OESv3bChAmQJAn379/XakyvqjTi0fQ9379/PyRJwv79+7W27/Jo3LhxaNasma7DoFLCJF/Grl27hkGDBqF69eowMzODjY0N/P39MWfOHGRnZ5fqvsPCwnDhwgVMmTIFK1euRJMmTUp1f2UpPDwckiTBxsamyPfxypUrkCQJkiTh22+/LXH/d+7cwYQJE3D27FktRPt6Jk2ahGbNmsHf31+VqDRZSD8UFBRg+vTp8PT0hJmZGerXr481a9a8Ul8DBgyAJEl47733ily/detWNGrUCGZmZnB3d8dXX32FJ0+eqG0zYsQInDt3Dlu3bn2lGEi/Ges6AEOyfft2dOvWDQqFAr1790bdunWRm5uLQ4cOYcyYMbh06RIWL15cKvvOzs7G0aNHMX78eAwZMqRU9uHh4YHs7GyYmJiUSv8vY2xsjKysLGzbtg0ffPCB2rpVq1bBzMwMOTk5r9T3nTt3MHHiRFSrVg0NGzbU+HV//vnnK+2vOPfu3cPy5cuxfPlyAICPjw9Wrlyptk1UVBSsrKwwfvx4re6btGP8+PGYNm0aBgwYgKZNm+K3335Dz549IUkSunfvrnE/p06dwrJly2BmZlbk+p07d6Jz585o2bIl5s6diwsXLuDrr7/G3bt3sWDBAtV2zs7OCAkJwbfffotOnTq99vGRfmGSLyMJCQno3r07PDw8EBMTAxcXF9W6iIgIXL16Fdu3by+1/d+7dw8AYGdnV2r7kCSp2D84ZUGhUMDf3x9r1qwplORXr16Njh07YuPGjWUSS1ZWFiwsLGBqaqrVfn/55RcYGxsjODgYAODk5IRevXqpbTNt2jRUqlSpUPvrKigoQG5urk7/H5d3t2/fxsyZMxEREYF58+YBAPr374+AgACMGTMG3bp1Q4UKFV7ajxACw4YNQ+/evbF3794itxk9ejTq16+PP//8E8bGT//U29jYYOrUqRg+fDhq166t2vaDDz5At27dcP36dVSvXl0LR0r6guX6MjJ9+nRkZGTg559/Vkvwz3h5eWH48OGqn588eYLJkyejRo0aUCgUqFatGj777DMolUq111WrVg3vvfceDh06hDfffBNmZmaoXr06VqxYodpmwoQJ8PDwAACMGTMGkiShWrVqAJ6WuZ/9+3nPzpU+b/fu3Xj77bdhZ2cHKysreHt747PPPlOtL+6cfExMDN555x1YWlrCzs4OISEhiIuLK3J/V69eRXh4OOzs7GBra4s+ffogKyur+Df2P3r27ImdO3ciNTVV1Xby5ElcuXIFPXv2LLT9w4cPMXr0aNSrVw9WVlawsbFB+/btce7cOdU2+/fvR9OmTQEAffr0UZW/nx1ny5YtUbduXcTGxqJFixawsLBQvS//PT8cFhYGMzOzQsfftm1b2Nvb486dOy88vi1btqBZs2awsrLS+D0pSmpq6kvfZ0mSMGTIEKxatQp16tSBQqHArl27ADxNVn379oWTkxMUCgXq1KmDJUuWFNrP3LlzUadOHVhYWMDe3h5NmjTB6tWrXykeTT8TRbl16xY6d+4MS0tLVK5cGZGRkRq9Ttt+++035OXl4ZNPPlG1SZKEwYMH49atWzh69KhG/axcuRIXL17ElClTilz/999/4++//8bAgQNVCR4APvnkEwghsGHDBrXtg4KCVPGRvHAkX0a2bduG6tWr46233tJo+/79+2P58uXo2rUrRo0ahePHjyM6OhpxcXHYvHmz2rZXr15F165d0a9fP4SFhWHJkiUIDw9H48aNUadOHXTp0gV2dnaIjIxEjx490KFDhxIniUuXLuG9995D/fr1MWnSJCgUCly9evWlk7/27NmD9u3bo3r16pgwYQKys7Mxd+5c+Pv74/Tp04W+YHzwwQfw9PREdHQ0Tp8+jZ9++gmVK1fGN998o1GcXbp0wccff4xNmzah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QM+ePTFhwgRERESgXr16aufhz507h6tXr8La2lqtj5ycHFy7dg1paWlISkpCs2bNVOuMjY3RpEmTQiX7Z86ePYsKFSogICBA45ivXr2KrKwsvPvuu2rtubm5eOONNwAAcXFxanEAgJ+fn8b7ICLdY5Inek2BgYFYsGABTE1N4erqCmPj//tYWVpaqm2bkZGBxo0bY9WqVYX6cXR0fKX9m5ubl/g1GRkZAIDt27ejSpUqausUCsUrxUFE+odJnug1WVpawsvLS6NtGzVqhF9//RWVK1eGjY1Nkdu4uLjg+PHjaNGiBQDgyZMniI2NRaNGjYrcvl69eigoKMCBAwcQFBRUaP2zSkJ+fr6qzdfXFwqFAomJicVWAHx8fLB161a1tmPHjr38IIlIb3DiHVEZ+uijj1CpUiWEhITgr7/+QkJCAvbv349hw4bh1q1bAIDhw4dj2rRp2LJlC/755x988sknL7zGvVq1aggLC0Pfvn2xZcsWVZ/r1q0DAHh4eECSJPz++++4d+8eMjIyYG1tjdGjRyMyMhLLly/HtWvXcPr0acydOxfLly8HAHz88ce4cuUKxowZg/j4eKxevRrLli0r7beIiLSISZ6oDFlYWODgwYNwd3dHly5d4OPjg379+iEnJ0c1sh81ahT+97//ISwsDH5+frC2tsb777//wn4XLFiArl274pNPPkHt2rUxYMAAZGZmAgCqVKmCiRMnYty4cXBycsKQIUMAAJMnT8YXX3yB6Oho+Pj4oF27dti+fTs8PT0BAO7u7ti4cSO2bNmCBg0aYOHChZg6dWopvjtEpG2SKG42DxEREZVrHMkTERHJFJM8ERGRTDHJExERyRSTPBERkUwxyRMREckUkzwREZFMMckTERHJFJM8ERGRTDHJExERyRSTPBERkUwxyRMREcnU/wMP8g18tDDSLAAAAABJRU5ErkJggg=="/>
</div>
</div>
<div class="jp-OutputArea-child">
<div class="jp-OutputPrompt jp-OutputArea-prompt"></div>
<div class="jp-RenderedImage jp-OutputArea-output" tabindex="0">
<img alt="No description has been provided for this image" class="" src="data:image/png;base64,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"/>
</div>
</div>
</div>
</div>
</div>
</main>
</body>
</html>
