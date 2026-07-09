import { apiFetch } from '$lib/utils';
import { API_MOE } from '$lib/constants';
import type { ApiMoeRoutedExpertsCompletionResponse } from '$lib/types/api';

export class MoeService {
	/**
	 * Fetch aggregated MoE routed expert data for a completed chat completion.
	 *
	 * @param completionId - The chat completion id returned in the streaming response
	 * @returns Aggregated expert counts per layer, or null if no data is available
	 */
	static async fetchRoutedExpertsForCompletion(
		completionId: string
	): Promise<ApiMoeRoutedExpertsCompletionResponse | null> {
		if (!completionId) {
			return null;
		}

		const url = `${API_MOE.ROUTED_EXPERTS_COMPLETION}?completion_id=${encodeURIComponent(completionId)}`;

		try {
			const response = await fetch(url);

			if (response.status === 404) {
				return null;
			}

			if (!response.ok) {
				const error = await response.text();
				console.error(`Failed to fetch MoE expert data: ${error}`);
				return null;
			}

			return (await response.json()) as ApiMoeRoutedExpertsCompletionResponse;
		} catch (error) {
			console.error('Failed to fetch MoE expert data:', error);
			return null;
		}
	}
}
