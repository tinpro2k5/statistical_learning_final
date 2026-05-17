export async function searchDocumentsRequest(context, keywords, nResults, nlp_server_address, timeout = 200000) {
  const data = {
    ranking_variable: `[AI]${context}`,
    keywords: keywords.replaceAll(';', '<AND>'),
    paper_list: '',
    prefetch_nResults_per_collection: 100,
    nResults: nResults,
    requires_removing_duplicates: true,
    requires_additional_prefetching: false,
    requires_reranking: true,
    reranking_method: 'scibert',
  };

  const controller = new AbortController();
  const timeout_id = setTimeout(() => controller.abort(), timeout);
  const requestOptions = {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
    signal: controller.signal,
  };
  const response = await fetch(`${nlp_server_address}/ml-api/doc-search/v1.0`, requestOptions);
  const response_data = await response.json();
  clearTimeout(timeout_id);
  return response_data['response'];
}

export async function get_papers_content(paper_list, nlp_server_address, timeout = 200000) {
  const data = {
    paper_list: paper_list,
    projection: null,
  };

  const controller = new AbortController();
  const timeout_id = setTimeout(() => controller.abort(), timeout);
  const requestOptions = {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
    signal: controller.signal,
  };
  const response = await fetch(`${nlp_server_address}/ml-api/get-papers/v1.0`, requestOptions);
  const response_data = await response.json();
  clearTimeout(timeout_id);
  return response_data['response'];
}

export async function generate_citations_for_papers(paper_content_list, context, keywords, nlp_server_address, timeout = 200000) {
  const context_list = paper_content_list.map(() => context);
  const keywords_list = paper_content_list.map(() => keywords.replaceAll(';', '\t'));
  const papers = paper_content_list.map((item) => ({
    Title: item.Title ?? '',
    Abstract: item.Abstract ?? '',
  }));

  const data = {
    context_list: context_list,
    keywords_list: keywords_list,
    papers: papers,
  };

  const controller = new AbortController();
  const timeout_id = setTimeout(() => controller.abort(), timeout);
  const requestOptions = {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
    signal: controller.signal,
  };
  const response = await fetch(`${nlp_server_address}/ml-api/generate-citation/v1.0`, requestOptions);
  const response_data = await response.json();
  clearTimeout(timeout_id);
  const response_data_response = response_data['response'];
  return response_data_response.map((item) => item.trim());
}
