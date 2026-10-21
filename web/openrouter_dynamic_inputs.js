/**
 * Dynamic inputs for OpenRouter Nodes
 * - Handles dynamic image_N slot management for text LLM and image generation nodes
 * - Handles dynamic voice dropdown updates for speech node based on selected model
 */

import { app } from "../../../scripts/app.js"

const TypeSlot = {
    Input: 1,
    Output: 2,
};

const TypeSlotEvent = {
    Connect: true,
    Disconnect: false,
};

// Node types that get dynamic image input slots
const DYNAMIC_IMAGE_NODE_IDS = new Set([
    "OpenRouterNode",
    "OpenRouterImageGenNode",
    "OpenRouterImageAPINode",
    "OpenRouterDecisionNode",
]);

// Node types that get dynamic voice dropdown
const DYNAMIC_VOICE_NODE_IDS = new Set([
    "OpenRouterSpeechNode",
]);

const PREFIX = "image";
const TYPE = "IMAGE";

// The api_key widget is a hidden placeholder that keeps saved widgets_values aligned.
// The key itself is read by the server from LLM_KEY or openrouter_api_key.json.
app.registerExtension({
    name: 'OpenRouter.HiddenApiKey',
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (!nodeData.name.startsWith("OpenRouter") || !nodeData.input?.required?.api_key) {
            return
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const me = onNodeCreated?.apply(this);
            const apiKeyWidget = this.widgets.find(w => w.name === "api_key");
            apiKeyWidget.hidden = true;
            apiKeyWidget.serializeValue = () => "";
            apiKeyWidget.callback = () => { apiKeyWidget.value = "" };
            this.setSize([this.size[0], this.computeSize()[1]]);
            return me;
        }

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const me = onConfigure?.apply(this, arguments);
            const apiKeyWidget = this.widgets.find(w => w.name === "api_key");
            this.disconnectInput("api_key");
            if (apiKeyWidget.value) {
                console.warn("[OpenRouter] Ignored the api_key saved in this workflow. Keys are read only from the LLM_KEY env var or openrouter_api_key.json.");
                apiKeyWidget.value = "";
            }
            return me;
        }
    },
})

app.registerExtension({
    name: 'OpenRouter.DynamicImageInputs',
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        // Skip if not one of our nodes
        if (!DYNAMIC_IMAGE_NODE_IDS.has(nodeData.name)) {
            return
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const me = onNodeCreated?.apply(this);

            // Start with a new dynamic input slot
            this.addInput(PREFIX, TYPE);

            // Ensure the new slot has proper appearance
            const slot = this.inputs[this.inputs.length - 1];
            if (slot) {
                slot.color_off = "#666";
            }

            return me;
        }

        const onConnectionsChange = nodeType.prototype.onConnectionsChange
        nodeType.prototype.onConnectionsChange = function (slotType, slot_idx, event, link_info, node_slot) {
            const me = onConnectionsChange?.apply(this, arguments);

            // Links are restored one at a time while a workflow loads; cleaning up now
            // would delete image slots whose links haven't been reattached yet.
            if (app.configuringGraph) {
                return me;
            }

            if (slotType === TypeSlot.Input) {
                // Only process image inputs
                if (node_slot && !node_slot.name.startsWith(PREFIX)) {
                    return me;
                }

                if (link_info && event === TypeSlotEvent.Connect) {
                    // Get the parent (left side node) from the link
                    const fromNode = this.graph._nodes.find(
                        (otherNode) => otherNode.id == link_info.origin_id
                    )

                    if (fromNode) {
                        // Make sure there is a parent for the link
                        const parent_link = fromNode.outputs[link_info.origin_slot];
                        if (parent_link) {
                            node_slot.type = parent_link.type;
                            node_slot.name = `${PREFIX}_`;
                        }
                    }
                } else if (event === TypeSlotEvent.Disconnect) {
                    // Don't remove the slot immediately, let the cleanup below handle it
                }

                // Track each slot name so we can index the uniques
                let idx = 0;
                let slot_tracker = {};
                let toRemove = [];

                for(const slot of this.inputs) {
                    // Skip non-image inputs
                    if (!slot.name.startsWith(PREFIX)) {
                        idx += 1;
                        continue;
                    }

                    // Mark empty image slots for removal (except the last one)
                    if (slot.link === null && idx < this.inputs.length - 1) {
                        toRemove.push(idx);
                    } else if (slot.link !== null) {
                        // Connected slot - update its name with proper index
                        const name = slot.name.split('_')[0];
                        let count = (slot_tracker[name] || 0) + 1;
                        slot_tracker[name] = count;
                        slot.name = `${name}_${count}`;
                    }
                    idx += 1;
                }

                // Remove empty slots from highest index to lowest
                toRemove.reverse();
                for(const removeIdx of toRemove) {
                    this.removeInput(removeIdx);
                }

                // Check if the last input is an image input
                let lastInput = null;
                for (let i = this.inputs.length - 1; i >= 0; i--) {
                    if (this.inputs[i].name.startsWith(PREFIX)) {
                        lastInput = this.inputs[i];
                        break;
                    }
                }

                // If there's no empty image slot at the end, or no image slots at all, add one
                if (!lastInput || lastInput.link !== null) {
                    this.addInput(PREFIX, TYPE);
                    // Set the unconnected slot to appear gray
                    const newSlot = this.inputs[this.inputs.length - 1];
                    if (newSlot) {
                        newSlot.color_off = "#666";
                    }
                }

                // Force the node to resize itself for the new/deleted connections
                this?.graph?.setDirtyCanvas(true);
                return me;
            }
        }

        return nodeType;
    },
})

// Dynamic voice dropdown for SpeechNode based on model selection
app.registerExtension({
    name: 'OpenRouter.DynamicVoices',
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        // Only apply to speech node
        if (!DYNAMIC_VOICE_NODE_IDS.has(nodeData.name)) {
            return
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const me = onNodeCreated?.apply(this);

            // Find the model and voice widgets
            const modelWidget = this.widgets?.find(w => w.name === "model");
            const voiceWidget = this.widgets?.find(w => w.name === "voice");

            if (modelWidget && voiceWidget) {
                console.log("[SpeechNode] Found model and voice widgets");

                // Store original callback
                const originalModelCallback = modelWidget.callback;

                // Override model widget callback to update voices when model changes
                modelWidget.callback = async (value) => {
                    console.log(`[SpeechNode] Model changed to: ${value}`);

                    // Call original callback if it exists
                    if (originalModelCallback) {
                        originalModelCallback.apply(modelWidget, arguments);
                    }

                    // Fetch voices for the selected model
                    try {
                        const url = "/openrouter/voices/" + encodeURIComponent(value);
                        console.log(`[SpeechNode] Fetching voices from: ${url}`);

                        const response = await fetch(url);

                        if (response.ok) {
                            const data = await response.json();
                            const voices = data.voices || [];

                            console.log(`[SpeechNode] Got voices: ${JSON.stringify(voices)}`);

                            // Update voice widget options - ComfyUI expects options.values for combos
                            if (Array.isArray(voices) && voices.length > 0) {
                                // For ComfyUI combo widgets, update the options array
                                voiceWidget.options.values = voices;

                                // Set to first available voice if current selection isn't available
                                if (!voices.includes(voiceWidget.value)) {
                                    voiceWidget.value = voices[0];
                                    console.log(`[SpeechNode] Set voice to: ${voices[0]}`);
                                } else {
                                    console.log(`[SpeechNode] Kept voice: ${voiceWidget.value}`);
                                }

                                // Mark canvas dirty to force UI update
                                if (this.graph) {
                                    this.graph.setDirtyCanvas(true);
                                }
                            } else {
                                console.log(`[SpeechNode] No supported voices list for model ${value}`);
                            }
                        } else {
                            console.warn(`[SpeechNode] Failed to fetch voices - status ${response.status}`);
                            const text = await response.text();
                            console.warn(`[SpeechNode] Response: ${text}`);
                        }
                    } catch (error) {
                        console.warn(`[SpeechNode] Error fetching voices: ${error}`);
                    }
                };
            }

            return me;
        }

        return nodeType;
    },
})

// Decision node shows only the criteria fields that apply to the selected question type
app.registerExtension({
    name: 'OpenRouter.DecisionQuestionType',
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "OpenRouterDecisionNode") {
            return
        }

        const showFieldsForType = (node) => {
            const noul = node.widgets.find(w => w.name === "question_type").value === "noul";
            for (const widget of node.widgets) {
                if (widget.name === "true_guidance" || widget.name === "false_guidance") {
                    widget.hidden = !noul;
                } else if (widget.name === "criteria") {
                    widget.hidden = noul;
                }
            }
            node.setSize([node.size[0], node.computeSize()[1]]);
        };

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const me = onNodeCreated?.apply(this);
            this.widgets.find(w => w.name === "question_type").callback = () => showFieldsForType(this);
            showFieldsForType(this);
            return me;
        }

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const me = onConfigure?.apply(this, arguments);
            showFieldsForType(this);
            return me;
        }
    },
})
