import { app } from "/scripts/app.js";

app.registerExtension({
    name: "Comfyui_Qwen3-VL-Instruct-Plus.MultiplePathsInputPlus",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (!nodeData?.category?.startsWith("Comfyui_Qwen3-VL-Instruct-Plus")) {
            return;
        }
        switch (nodeData.name) {
            case "MultiplePathsInputPlus":
                nodeType.prototype.onNodeCreated = function () {
                    this._type = "PATH";
                    this.inputs_offset = nodeData.name.includes("selective") ? 1 : 0;

                    // 添加输入数量控制参数
                    this.addWidget("number", "inputcount", 1, (v) => {
                        v = Math.max(1, Math.min(1000, Math.round(v)));
                        return v;
                    }, { min: 1, max: 1000, step: 1 });

                    // 修复输入框更新逻辑（核心）
                    this.addWidget("button", "Update inputs", null, () => {
                        if (!this.inputs) {
                            this.inputs = [];
                        }

                        // 获取目标输入数量
                        const inputcountWidget = this.widgets.find(w => w.name === "inputcount");
                        if (!inputcountWidget) return;
                        const target_number_of_inputs = inputcountWidget.value;

                        // 仅统计以"path_"开头的有效输入
                        const currentPathInputs = this.inputs.filter(input => 
                            input.name?.startsWith("path_")
                        );
                        const currentCount = currentPathInputs.length;

                        if (target_number_of_inputs === currentCount) return;

                        // 移除多余输入
                        if (target_number_of_inputs < currentCount) {
                            for (let i = currentCount; i > target_number_of_inputs; i--) {
                                const inputToRemove = this.inputs.find(
                                    input => input.name === `path_${i}`
                                );
                                if (inputToRemove) {
                                    this.removeInput(this.inputs.indexOf(inputToRemove));
                                }
                            }
                        } 
                        // 添加缺少的输入
                        else {
                            for (let i = currentCount + 1; i <= target_number_of_inputs; i++) {
                                this.addInput(`path_${i}`, this._type);
                            }
                        }
                    });

                    // 关键修复：初始只添加1个path_1（避免重复）
                    // 先检查是否已有path_1，没有则添加
                    const hasPath1 = this.inputs?.some(input => input.name === "path_1");
                    if (!hasPath1) {
                        this.addInput("path_1", this._type);
                    }
                };
                break;
        }
    },
    async setup() {
        const originalComputeVisibleNodes =
            LGraphCanvas.prototype.computeVisibleNodes;
        LGraphCanvas.prototype.computeVisibleNodes = function () {
            const visibleNodesSet = new Set(
                originalComputeVisibleNodes.apply(this, arguments)
            );
            for (const node of this.graph._nodes) {
                if (
                    (node.type === "SetNode" || node.type === "GetNode") &&
                    node.drawConnection
                ) {
                    visibleNodesSet.add(node);
                }
            }
            return Array.from(visibleNodesSet);
        };
    },
});