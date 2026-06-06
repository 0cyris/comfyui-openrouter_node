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
]);

// Node types that get dynamic voice dropdown
const DYNAMIC_VOICE_NODE_IDS = new Set([
    "OpenRouterSpeechNode",
]);

const PREFIX = "image";
const TYPE = "IMAGE";

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

            // api_key security label — only on the text LLM node
            if (nodeData.name === "OpenRouterNode") {
                const apiKeyWidget = this.widgets?.find(w => w.name === "api_key");
                if (apiKeyWidget) {
                    apiKeyWidget.label = "api_key (Leave blank for secure loading)";
                }
            }

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
                // Store original callback
                const originalModelCallback = modelWidget.callback;

                // Override model widget callback to update voices when model changes
                modelWidget.callback = async (value) => {
                    // Call original callback if it exists
                    if (originalModelCallback) {
                        originalModelCallback.apply(modelWidget, arguments);
                    }

                    // Fetch voices for the selected model
                    try {
                        const apiKeyWidget = this.widgets?.find(w => w.name === "api_key");
                        const apiKey = apiKeyWidget?.value || "";

                        // Call the backend API to get supported voices
                        const response = await fetch("/openrouter/voices/" + encodeURIComponent(value), {
                            method: "GET",
                            headers: apiKey ? { "Authorization": `Bearer ${apiKey}` } : {},
                        });

                        if (response.ok) {
                            const data = await response.json();
                            const voices = data.voices || [];

                            // Update voice widget options
                            if (Array.isArray(voices) && voices.length > 0) {
                                voiceWidget.options.content = voices;
                                // Set to first available voice if current selection isn't available
                                if (!voices.includes(voiceWidget.value)) {
                                    voiceWidget.value = voices[0];
                                }
                                console.log(`[SpeechNode] Updated voices for model ${value}: ${voices.join(", ")}`);
                            } else {
                                // If no specific voices, allow free text
                                console.log(`[SpeechNode] No supported voices list for model ${value}, voice remains free-text`);
                            }
                        } else {
                            console.warn(`[SpeechNode] Failed to fetch voices for model ${value}`);
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
