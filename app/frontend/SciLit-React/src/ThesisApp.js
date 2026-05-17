import React, { useState } from 'react';
import {
  Alert,
  Box,
  Button,
  Card,
  CardContent,
  Chip,
  CircularProgress,
  Divider,
  Paper,
  Stack,
  Tab,
  Tabs,
  TextField,
  Typography,
} from '@mui/material';

import { get_papers_content, generate_citations_for_papers, searchDocumentsRequest } from './modules/utils';
import './ThesisApp.css';

const serverAddress = process.env.REACT_APP_NLP_SERVER_ADDRESS || 'http://localhost:8060';

function ResultCard({ paper, rank }) {
  const abstract = paper?.Content?.Abstract || paper?.Abstract || 'No abstract available.';
  return (
    <Card className="thesis-result-card" variant="outlined">
      <CardContent>
        <Stack spacing={1}>
          <Stack direction="row" spacing={1} alignItems="center" flexWrap="wrap">
            <Chip label={`#${rank}`} size="small" color="primary" />
            <Typography variant="h6" className="thesis-card-title">
              {paper?.Title || 'Untitled paper'}
            </Typography>
          </Stack>
          <Typography variant="body2" className="thesis-muted">
            {paper?.Venue || paper?.PublicationVenue || paper?.Year || ''}
          </Typography>
          <Typography variant="body2" className="thesis-abstract">
            {abstract}
          </Typography>
        </Stack>
      </CardContent>
    </Card>
  );
}

export default function ThesisApp() {
  const [tab, setTab] = useState(0);

  const [recommendContext, setRecommendContext] = useState('Recent progress in retrieval augmented generation and scientific literature search.');
  const [recommendKeywords, setRecommendKeywords] = useState('Transformer; retrieval; ranking');
  const [recommendLimit, setRecommendLimit] = useState(5);
  const [recommendLoading, setRecommendLoading] = useState(false);
  const [recommendError, setRecommendError] = useState('');
  const [recommendResults, setRecommendResults] = useState([]);

  const [citationContext, setCitationContext] = useState('The following sentence needs a citation.');
  const [citationKeywords, setCitationKeywords] = useState('Transformer; citation generation');
  const [citationTitle, setCitationTitle] = useState('A Transformer-based model for citation generation');
  const [citationAbstract, setCitationAbstract] = useState('This paper studies ...');
  const [citationLoading, setCitationLoading] = useState(false);
  const [citationError, setCitationError] = useState('');
  const [citationText, setCitationText] = useState('');

  const handleSearch = async () => {
    setRecommendLoading(true);
    setRecommendError('');
    try {
      const paperIds = await searchDocumentsRequest(
        recommendContext,
        recommendKeywords,
        Number(recommendLimit),
        serverAddress,
      );
      const paperInfoList = await get_papers_content(paperIds, serverAddress);
      setRecommendResults(Array.isArray(paperInfoList) ? paperInfoList : []);
    } catch (error) {
      setRecommendResults([]);
      setRecommendError(error?.message || 'Unable to fetch recommendations.');
    } finally {
      setRecommendLoading(false);
    }
  };

  const handleGenerateCitation = async () => {
    setCitationLoading(true);
    setCitationError('');
    setCitationText('');
    try {
      const generated = await generate_citations_for_papers(
        [{ Title: citationTitle, Abstract: citationAbstract }],
        citationContext,
        citationKeywords,
        serverAddress,
      );
      setCitationText((generated && generated[0]) || 'No output returned.');
    } catch (error) {
      setCitationError(error?.message || 'Unable to generate citation.');
    } finally {
      setCitationLoading(false);
    }
  };

  return (
    <Box className="thesis-shell">
      <Box className="thesis-hero">
        <Typography variant="overline" className="thesis-eyebrow">
          Transformer-based NLP demo
        </Typography>
        <Typography variant="h3" className="thesis-title">
          SciLit focused app
        </Typography>
        <Typography variant="body1" className="thesis-subtitle">
          A compact front end for the two Transformer cores you selected: paper recommendation and citation generation.
        </Typography>
      </Box>

      <Paper className="thesis-panel" elevation={0}>
        <Tabs value={tab} onChange={(event, value) => setTab(value)} className="thesis-tabs">
          <Tab label="Paper Recommendation" />
          <Tab label="Citation Generation" />
        </Tabs>

        <Divider />

        {tab === 0 && (
          <Box className="thesis-layout">
            <Stack spacing={2} className="thesis-form-column">
              <TextField
                label="Research context"
                value={recommendContext}
                onChange={(event) => setRecommendContext(event.target.value)}
                multiline
                minRows={6}
                fullWidth
              />
              <TextField
                label="Keywords"
                value={recommendKeywords}
                onChange={(event) => setRecommendKeywords(event.target.value)}
                fullWidth
              />
              <TextField
                label="Top N results"
                type="number"
                value={recommendLimit}
                onChange={(event) => setRecommendLimit(event.target.value)}
                fullWidth
              />
              <Button variant="contained" size="large" onClick={handleSearch} disabled={recommendLoading}>
                {recommendLoading ? 'Searching...' : 'Recommend papers'}
              </Button>
              {recommendError ? <Alert severity="error">{recommendError}</Alert> : null}
            </Stack>

            <Box className="thesis-results-column">
              <Typography variant="h6" className="thesis-section-title">
                Ranked papers
              </Typography>
              {recommendLoading ? (
                <Box className="thesis-center">
                  <CircularProgress />
                </Box>
              ) : null}
              <Stack spacing={2}>
                {recommendResults.map((paper, index) => (
                  <ResultCard key={`${paper?._id || index}`} paper={paper} rank={index + 1} />
                ))}
                {!recommendLoading && recommendResults.length === 0 ? (
                  <Alert severity="info">Run a search to see the reranked paper list.</Alert>
                ) : null}
              </Stack>
            </Box>
          </Box>
        )}

        {tab === 1 && (
          <Box className="thesis-layout">
            <Stack spacing={2} className="thesis-form-column">
              <TextField
                label="Citation context"
                value={citationContext}
                onChange={(event) => setCitationContext(event.target.value)}
                multiline
                minRows={4}
                fullWidth
              />
              <TextField
                label="Keywords"
                value={citationKeywords}
                onChange={(event) => setCitationKeywords(event.target.value)}
                fullWidth
              />
              <TextField
                label="Cited paper title"
                value={citationTitle}
                onChange={(event) => setCitationTitle(event.target.value)}
                fullWidth
              />
              <TextField
                label="Cited paper abstract"
                value={citationAbstract}
                onChange={(event) => setCitationAbstract(event.target.value)}
                multiline
                minRows={5}
                fullWidth
              />
              <Button variant="contained" size="large" onClick={handleGenerateCitation} disabled={citationLoading}>
                {citationLoading ? 'Generating...' : 'Generate citation'}
              </Button>
              {citationError ? <Alert severity="error">{citationError}</Alert> : null}
            </Stack>

            <Box className="thesis-results-column">
              <Typography variant="h6" className="thesis-section-title">
                Generated citation
              </Typography>
              {citationLoading ? (
                <Box className="thesis-center">
                  <CircularProgress />
                </Box>
              ) : null}
              {citationText ? (
                <Paper className="thesis-output-box" variant="outlined">
                  <Typography variant="body1">{citationText}</Typography>
                </Paper>
              ) : null}
              {!citationLoading && !citationText ? (
                <Alert severity="info">Generate a citation sentence to preview the model output.</Alert>
              ) : null}
            </Box>
          </Box>
        )}
      </Paper>
    </Box>
  );
}
