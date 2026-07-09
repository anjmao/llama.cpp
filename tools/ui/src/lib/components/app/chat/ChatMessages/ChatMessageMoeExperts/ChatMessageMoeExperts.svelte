<script lang="ts">
	import * as Tooltip from '$lib/components/ui/tooltip';
	import type { ApiMoeRoutedExpertsCompletionResponse } from '$lib/types/api';

	interface Props {
		data: ApiMoeRoutedExpertsCompletionResponse;
	}

	let { data }: Props = $props();

	function maxCountForLayer(counts: number[]): number {
		return Math.max(1, ...counts);
	}

	function heatmapColor(ratio: number): string {
		// blend from muted background to primary-ish blue
		const r = Math.round(24 + (59 - 24) * ratio);
		const g = Math.round(24 + (130 - 24) * ratio);
		const b = Math.round(27 + (246 - 27) * ratio);
		return `rgb(${r} ${g} ${b})`;
	}
</script>

<div class="space-y-2 text-xs">
	<div class="flex items-center justify-between text-muted-foreground">
		<span>Completion: {data.completion_id}</span>
		<span>{data.layers.length} layers x {data.n_experts} experts</span>
	</div>

	<div class="flex flex-col gap-1 overflow-x-auto pb-1">
		{#each data.layers as layer (layer.layer)}
			{@const maxCount = maxCountForLayer(layer.counts)}
			<div class="flex items-center gap-2">
				<span class="w-10 shrink-0 text-right tabular-nums text-muted-foreground">
					L{layer.layer}
				</span>
				<div class="flex gap-px">
					{#each layer.counts as count, expert (expert)}
						{@const ratio = count / maxCount}
						{@const percentage = layer.total > 0 ? (count / layer.total) * 100 : 0}
						<Tooltip.Root>
							<Tooltip.Trigger>
								{#snippet child({ props })}
									<div
										{...props}
										class="h-4 w-4 rounded-sm"
										style="background-color: {heatmapColor(ratio)}; opacity: {count > 0 ? 0.4 + 0.6 * ratio : 0.2};"
										role="img"
										aria-label="Layer {layer.layer}, expert {expert}: {count} ({percentage.toFixed(1)}%)"
									></div>
								{/snippet}
							</Tooltip.Trigger>
							<Tooltip.Content>
								<p>Layer {layer.layer}, expert {expert}</p>
								<p>{count} selections ({percentage.toFixed(1)}%)</p>
							</Tooltip.Content>
						</Tooltip.Root>
					{/each}
				</div>
			</div>
		{/each}
	</div>
</div>
