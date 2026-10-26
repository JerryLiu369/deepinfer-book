# 06 示范稿：第 6 章第 2 节

> **给主代理的说明**
>
> 这是目标文风的样板。它展示了：开头先给结论；"问题 → 思路 → 做法 → 代价 → 对比"的结构（小标题写得自然）；术语首次出现时的解释；数字带条件；代码引用的方式；证据等级的措辞；不确定内容的标记方式。
>
> 示范稿中的事实来自 `doc/inference_deployment_qa.md` 和 `9bd9cf2` 版本源码。正式写作时仍须用 R05 的调研结果核对、补充。文中 `【待核实：……】` 是**写作过程中**的标记方式，最终书稿里不能留下。
>
> 篇幅约 2,500 字。正式章节中，一节一般在 1,500–3,500 字之间。

---

## 6.2 异步调度：别让 GPU 等 CPU

DeepInfer 让 CPU 提前一步准备下一批工作，GPU 算完一步就能马上开始下一步，中间不再空等。这个机制在代码里叫 ZOS（zero-overhead scheduling，零开销调度），默认开启。它不是 DeepInfer 首创：DeepSeek 在部署问答里明确说参考了 SGLang 和 NanoFlow 的做法。DeepInfer 的特别之处在于，它要让这套机制和投机解码（第 10 章）一起工作。

### GPU 在等什么

先看不做异步时，解码的一步（step）是怎么走的。每一步依次做四件事：

1. **调度（schedule）**：CPU 决定这一步有哪些请求参加，每个请求算几个 token。
2. **前向计算（forward）**：GPU 把这批 token 跑一遍模型。
3. **采样（sample）**：GPU 根据输出的概率分布，为每个请求选出下一个 token。
4. **后处理（post-process）**：CPU 把新 token 追加到各个请求上，检查哪些请求已经结束，释放它们占用的缓存。

问题出在第 3 步和第 4 步之间。CPU 要做后处理，就必须拿到采样结果，所以这里有一次 GPU 到 CPU 的同步：CPU 停下来等 GPU 把结果交回。同步完成后，CPU 先做第 N 步的后处理，再做第 N+1 步的调度，然后才能把新的计算任务交给 GPU。**这段时间里 GPU 无事可做。**

<!-- 图 6-2：同步调度时间线。用 mermaid gantt 画两条泳道：CPU 和 GPU。GPU 泳道在 post-process(N) 与 schedule(N+1) 期间留白，并标注"GPU 空闲"。 -->

**图 6-2 同步调度下的时间线。** 看 GPU 那一行的空白：每一步都有一段时间在等 CPU。

这段空白在 prefill 阶段不明显，因为一次 prefill 要算成千上万个 token，GPU 本身就要忙很久。但在 decode 阶段，每个请求每步只算几个 token，GPU 一步的计算时间很短，CPU 那部分开销所占的比例就变大了。DeepSeek 的文档没有给出这段空闲的实测时长。【待核实：R05 是否能从 `cpu_timer` 的统计项（如 `bubble`）或其他公开数据中找到量级；找不到就保留"文档没有给出"的说法。】

### 思路：让 CPU 跑在前面

既然 GPU 计算第 N 步的时候 CPU 闲着，那就让 CPU 利用这段时间去准备第 N+1 步。按 DeepSeek 的描述，开启异步调度后，GPU 执行 `forward(N) → sample(N)` 的同时，CPU 依次执行：

```
schedule(N+1) → launch_forward(N+1) → post-process(N−1) → launch_sample(N+1)
```

这条流水线有两个要点：

- CPU 在 GPU 还没算完第 N 步时，就已经把第 N+1 步的前向计算**排进了 GPU 的任务队列**（`launch_forward(N+1)`）。GPU 一做完第 N 步，立刻接着做第 N+1 步。
- CPU 做后处理的对象是**第 N−1 步**，不是第 N 步。也就是说，CPU 看到的结果始终比 GPU 慢一步。

<!-- 图 6-3：异步调度时间线。同样两条泳道，GPU 一行连续不断；CPU 一行标出 schedule(N+1)、post-process(N−1) 等块与 GPU 的 forward(N) 重叠。 -->

**图 6-3 异步调度下的时间线。** GPU 一行没有空白了；代价是 GPU 在算第 N 步时，CPU 才在处理第 N−1 步的结果。

代码里的注释直接写明了这个"慢一步"：

```python
# deepinfer/engine/llm_engine.py:824-828
def process_output(self, output: np.ndarray[np.uint64]) -> None:
    """虚函数，处理输出一个 step 输出的 token
    - ZOS=False: 处理当前 step
    - ZOS=True: 处理上一个 step 输出的 token
    """
```

这几行说明：同一个后处理函数，在关闭 ZOS 时处理当前步的输出，开启 ZOS 时处理上一步的输出。

### 难点：结果还没出来，下一步怎么排

"CPU 提前一步"听上去简单，难在 CPU 安排第 N+1 步时，**第 N 步的采样结果还在 GPU 上没出来**。至少有两个问题要解决。

**第一，下一步的输入 token 是什么？** 解码时，第 N+1 步的输入就是第 N 步刚采样出来的 token。CPU 不知道这个 token，就没法像同步模式那样把输入准备好再拷到 GPU。

DeepInfer 的做法是把"填入输入 token"这件事留在 GPU 上完成。`deepinfer/worker/zos_status.py` 中的 `MetadataZOS` 结构记录了需要的信息：

```python
# deepinfer/worker/zos_status.py:23-40（节选）
class MetadataZOS:
    remap: torch.Tensor | None
    """remap[i] = j if curr_batch[i] matches prev_batch[j], else -1"""
    prev_sampled_token_ids: torch.Tensor | None
    """[prev_n_seq], int32. prev_step sampled_token_ids"""
    prev_num_accepted_tokens: torch.Tensor | None
    """[prev_n_seq], int32. prev_step num_accepted_tokens, range [0, MTP_value]"""
```

- `remap` 是一张对照表：这一步的第 i 个请求，对应上一步的第 j 个请求；新加入的请求记为 −1。CPU 不知道 token 的值，但知道"谁是谁"，这张表 CPU 可以提前做好。
- `prev_sampled_token_ids` 是上一步的采样结果，它一直留在 GPU 显存里。GPU 按 `remap` 把它们取出来，填成这一步的输入。
- `prev_num_accepted_tokens` 是上一步投机解码中每个请求被接受的草稿 token 数。它决定了这一步每个请求的实际长度，所以长度信息也要在 GPU 上修正（同一结构中的 `seqlens_q`、`seqlens_k` 字段注释为 "corrected"）。

这样，CPU 只负责"安排谁参加、怎么对应"，具体数值由 GPU 自己补上，两边都不用等对方。【待核实：R05 确认 `remap` 由 `csrc/engine/zos.cpp` 生成，以及 GPU 端填充输入的具体函数位置。】

**第二，投机解码要验证几个 token？** 第 10 章会讲到，DeepInfer 用置信度来决定每个请求这一步验证多少个草稿 token。但在异步模式下，CPU 调度第 N+1 步时，第 N 步的置信度还没有返回。DeepSeek 的解决办法是：CPU 只确定这一步的**总预算**，每个请求分多少，由一个 TileLang 内核在 GPU 上重新计算，结果写回一块 pinned 内存（锁页内存，GPU 可以直接写入、CPU 可以直接读取的主机内存）。代码中对应的函数是 `compute_seqlens_q_by_confidence_rates`，定义在 `deepinfer/model_executor/layers/tilelang_ops/schedule_mtp_varlen.py`，由 `deepinfer/worker/worker.py` 调用。

两个问题的解法是同一个思路：**CPU 做不依赖结果的决定，依赖结果的部分挪到 GPU 上做。**

### 代价

异步调度不是免费的。

- **代码复杂度高。** 同一个请求在 CPU 和 GPU 上的"进度"不一致，资源释放也必须推迟。`llm_engine.py` 中专门有一个 `DeferredRelease` 类，注释写着"延后一轮释放 step 产出的资源，保证调用方读完"。引擎退出或清空时，还要用 `flush_inflight_reqs` 把"在途"的那一步处理完。
- **请求结束的感知晚一步。** 由于 CPU 处理的是上一步的结果，一个请求在第 N 步生成了结束符，CPU 要到更晚才知道。【待核实：R05 确认这种请求是否会被多调度一步，以及 DeepInfer 如何处理这一步的输出。】
- **调度决策用的是稍旧的信息。** 这正是投机解码长度要挪到 GPU 上重算的原因。

### vLLM 和 SGLang 怎么做

SGLang 较早推出了"零开销批调度器"（overlap scheduler），思路同样是让 CPU 调度与 GPU 计算重叠。vLLM 在 V1 引擎中也加入了异步调度选项。【待核实：R13、R14 给出两者的首次发布版本、当前是否默认开启、与投机解码能否同时使用。】三者思路相同，区别在于和各自其他机制的配合，例如 DeepInfer 要同时满足变长投机验证。第 14 章会把调度放进完整对比中。

### 小结

- 异步调度让 CPU 提前安排下一步，消除 decode 阶段 GPU 等待 CPU 的空闲。
- 代价是 CPU 永远晚一步看到结果。DeepInfer 的应对是：CPU 只做不依赖结果的决定，输入 token 和投机验证长度在 GPU 上补齐。
- 这项技术来自 SGLang 和 NanoFlow，DeepInfer 的工作重点在于让它与 DSpark 变长验证兼容。

下一节讲调度的另一半问题：prefill 时，每一步该放进多少 token。
