import React, { useState, useEffect, useRef, useCallback } from 'react';
import api from '@/api';
import logo from '@/assets/images/alkhidmat.png';
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer
} from 'recharts';

// ─── Colour palette ───────────────────────────────────────────────────────────
const COLORS = ['#3B82F6', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6'];

// ─── Eval constants ───────────────────────────────────────────────────────────
const METRIC_NAMES = [
  'AnswerRelevancy',
  'Faithfulness',
  'ContextualPrecision',
  'ContextualRecall',
  'Hallucination',
];

const METRIC_META = {
  AnswerRelevancy:    { short: 'AnsRel',  color: '#3B82F6', lowerBetter: false, threshold: 0.5  },
  Faithfulness:       { short: 'Faith',   color: '#10B981', lowerBetter: false, threshold: 0.5  },
  ContextualPrecision:{ short: 'CtxPre',  color: '#8B5CF6', lowerBetter: false, threshold: 0.3  },
  ContextualRecall:   { short: 'CtxRec',  color: '#F59E0B', lowerBetter: false, threshold: 0.3  },
  Hallucination:      { short: 'Halluc',  color: '#EF4444', lowerBetter: true,  threshold: 0.4  },
};

function isMetricPass(metric, score) {
  if (score === null || score === undefined) return null;
  const { lowerBetter, threshold } = METRIC_META[metric];
  return lowerBetter ? score <= threshold : score >= threshold;
}

function scoreColorClass(metric, score) {
  const pass = isMetricPass(metric, score);
  if (pass === null) return 'text-gray-400';
  return pass ? 'text-emerald-600 font-semibold' : 'text-red-500 font-semibold';
}

function computeEvalSummary(results) {
  const agg   = {};
  const pass  = {};
  METRIC_NAMES.forEach(m => { agg[m] = []; pass[m] = 0; });

  let rejections = 0, errors = 0, overallPass = 0;
  const scored = results.filter(r => !r.rag_rejected && !r.error);

  results.forEach(r => {
    if (r.rag_rejected) { rejections++; return; }
    if (r.error)        { errors++;     return; }
    if (r.overall_pass) overallPass++;
    METRIC_NAMES.forEach(m => {
      const s = r.scores?.[m];
      if (s !== null && s !== undefined) {
        agg[m].push(s);
        if (r.passed?.[m]) pass[m]++;
      }
    });
  });

  const avgs = {};
  METRIC_NAMES.forEach(m => {
    avgs[m] = agg[m].length ? agg[m].reduce((a, b) => a + b, 0) / agg[m].length : null;
  });

  return {
    avgs, pass, rejections, errors, overallPass,
    scoredCount: scored.length,
    total: results.length,
  };
}



// ─── Status badge ─────────────────────────────────────────────────────────────
function StatusBadge({ r }) {
  const { label, cls } =
    r.error        ? { label: 'CRASH',    cls: 'bg-gray-100 text-gray-600'    } :
    r.rag_rejected ? { label: 'REJECTED', cls: 'bg-amber-100 text-amber-700'  } :
    r.overall_pass ? { label: 'PASS',     cls: 'bg-emerald-100 text-emerald-700' } :
                     { label: 'FAIL',     cls: 'bg-red-100 text-red-700'      };
  return (
    <span className={`px-2 py-0.5 rounded-full text-xs font-semibold ${cls}`}>{label}</span>
  );
}

// ─── Expandable row detail ────────────────────────────────────────────────────
function RowDetail({ r }) {
  return (
    <tr className="bg-purple-50">
      <td colSpan={8} className="px-6 py-4">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
          <div>
            <p className="font-semibold text-gray-600 mb-1">Question</p>
            <p className="text-gray-800">{r.question}</p>
          </div>
          <div>
            <p className="font-semibold text-gray-600 mb-1">Expected Answer</p>
            <p className="text-gray-700 italic">{r.expected_answer}</p>
          </div>
          {r.actual_answer && (
            <div className="md:col-span-2">
              <p className="font-semibold text-gray-600 mb-1">Actual Answer</p>
              <p className="text-gray-800">{r.actual_answer}</p>
            </div>
          )}
          {r.reasons && Object.entries(r.reasons).some(([,v]) => v) && (
            <div className="md:col-span-2">
              <p className="font-semibold text-gray-600 mb-2">Metric Reasons</p>
              <div className="space-y-1">
                {METRIC_NAMES.map(m => r.reasons?.[m] && (
                  <div key={m} className="flex gap-2 text-xs">
                    <span className="font-medium text-gray-500 w-36 shrink-0">{m}</span>
                    <span className="text-gray-700">{r.reasons[m]}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
          {r.metadata && (
            <div>
              <p className="font-semibold text-gray-600 mb-1">RAG Metadata</p>
              <div className="text-xs text-gray-600 space-y-0.5">
                {r.rag_time_seconds !== undefined && <p>⏱ RAG time: {r.rag_time_seconds}s</p>}
                {r.contexts_retrieved !== undefined && <p>📄 Contexts retrieved: {r.contexts_retrieved}</p>}
                {r.metadata?.combined_confidence !== null && r.metadata?.combined_confidence !== undefined && (
                  <p>🎯 Combined confidence: {(r.metadata.combined_confidence * 100).toFixed(1)}%</p>
                )}
                {r.metadata?.domain && <p>🏷 Domain: {r.metadata.domain}</p>}
                {r.metadata?.selfrag_support && <p>✅ Self-RAG support: {r.metadata.selfrag_support}</p>}
              </div>
            </div>
          )}
        </div>
      </td>
    </tr>
  );
}

// ─── Evaluation Tab ───────────────────────────────────────────────────────────
const LANG_CONFIG = {
  english: { label: 'English',    flag: '🇬🇧', color: 'blue'   },
  urdu:    { label: 'Urdu',       flag: '🇵🇰', color: 'green'  },
  roman:   { label: 'Roman Urdu', flag: '📝',  color: 'purple' },
};

function EvalTab({ evalStatus, evalLang, evalData, evalLoading, evalError, onSelectLang }) {
  const currentData = evalData[evalLang] || null;

  const [expandedRow, setExpandedRow] = useState(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [statusFilter, setStatusFilter] = useState('all');

  const summary = currentData ? computeEvalSummary(currentData) : null;

  const barData = summary
    ? METRIC_NAMES.map(m => ({
        name: METRIC_META[m].short,
        score: summary.avgs[m] !== null ? parseFloat(summary.avgs[m].toFixed(3)) : 0,
        fill: METRIC_META[m].color,
      }))
    : [];

  const outcomeData = summary
    ? [
        { name: 'Pass',     value: summary.overallPass,                         fill: '#10B981' },
        { name: 'Fail',     value: summary.scoredCount - summary.overallPass,   fill: '#EF4444' },
        { name: 'Rejected', value: summary.rejections,                          fill: '#F59E0B' },
        { name: 'Error',    value: summary.errors,                              fill: '#9CA3AF' },
      ].filter(d => d.value > 0)
    : [];

  const filteredRows = currentData
    ? currentData.filter(r => {
        const matchSearch = !searchTerm || r.question?.toLowerCase().includes(searchTerm.toLowerCase());
        const status =
          r.error        ? 'crash'    :
          r.rag_rejected ? 'rejected' :
          r.overall_pass ? 'pass'     : 'fail';
        const matchStatus = statusFilter === 'all' || status === statusFilter;
        return matchSearch && matchStatus;
      })
    : [];

  return (
    <div>
      {/* ── Header + Language tabs ── */}
      <div className="mb-6">
        <h2 className="text-xl font-bold text-gray-800 mb-4">RAG Evaluation Reports</h2>
        <div className="flex gap-2 flex-wrap">
          {Object.entries(LANG_CONFIG).map(([lang, cfg]) => {
            const status = evalStatus[lang];
            const isActive = evalLang === lang;
            const hasData  = status?.exists;
            return (
              <button
                key={lang}
                onClick={() => onSelectLang(lang)}
                className={`flex items-center gap-2 px-5 py-2.5 rounded-xl font-semibold text-sm transition-all border-2 ${
                  isActive
                    ? 'bg-purple-600 text-white border-purple-600 shadow-md'
                    : hasData
                      ? 'bg-white text-gray-700 border-gray-200 hover:border-purple-400 hover:text-purple-700'
                      : 'bg-gray-50 text-gray-400 border-dashed border-gray-200 hover:border-gray-300'
                }`}
              >
                <span className="text-base">{cfg.flag}</span>
                <span>{cfg.label}</span>
                {hasData ? (
                  <span className={`text-xs px-1.5 py-0.5 rounded-full ${isActive ? 'bg-purple-500 text-white' : 'bg-emerald-100 text-emerald-700'}`}>
                    {status.total_cases} cases
                  </span>
                ) : (
                  <span className="text-xs px-1.5 py-0.5 rounded-full bg-gray-100 text-gray-400">no report</span>
                )}
              </button>
            );
          })}
        </div>
        {/* Last run info */}
        {evalStatus[evalLang]?.run_at && (
          <p className="text-xs text-gray-400 mt-2">
            Last run: {new Date(evalStatus[evalLang].run_at).toLocaleString()}
          </p>
        )}
      </div>

      {/* ── Empty state ── */}
      {!evalLoading && !currentData && (
        <div className="bg-white rounded-xl shadow p-14 text-center">
          <div className="text-5xl mb-4">{LANG_CONFIG[evalLang]?.flag}</div>
          <p className="text-lg font-semibold text-gray-700 mb-2">
            No {LANG_CONFIG[evalLang]?.label} report yet
          </p>
          <p className="text-sm text-gray-500 mb-1">Run this command on your server:</p>
          <code className="block bg-gray-100 text-purple-700 px-4 py-2 rounded-lg text-sm mt-2 mb-4">
            python test_rag_evaluation.py --language {evalLang} --use-openai-judge
          </code>
          <p className="text-xs text-gray-400">
            Results will be saved to <code>evaluation_results/report_{evalLang}.json</code> and appear here automatically.
          </p>
        </div>
      )}

      {evalLoading && (
        <div className="flex items-center justify-center py-20 text-gray-400">
          <svg className="animate-spin w-6 h-6 mr-3" fill="none" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8z"/>
          </svg>
          Loading report…
        </div>
      )}

      {evalError && (
        <div className="bg-red-50 border border-red-200 text-red-700 rounded-lg p-4 mb-4 text-sm">
          {evalError}
        </div>
      )}

      {summary && !evalLoading && (
        <>
          {/* ── Top stat cards ── */}
          <div className="grid grid-cols-2 md:grid-cols-4 xl:grid-cols-4 gap-4 mb-6">
            {[
              { label: 'Total Cases',      value: summary.total,                              color: 'text-gray-800'   },
              { label: 'Overall Pass',     value: `${summary.overallPass} / ${summary.scoredCount}`, color: 'text-emerald-600' },
              { label: 'Self-RAG Rejected',value: summary.rejections,                         color: 'text-amber-500'  },
              { label: 'Crashes / Errors', value: summary.errors,                             color: 'text-red-500'    },
            ].map(({ label, value, color }) => (
              <div key={label} className="bg-white rounded-xl shadow p-5">
                <p className="text-xs text-gray-500 mb-1">{label}</p>
                <p className={`text-3xl font-bold ${color}`}>{value}</p>
              </div>
            ))}
          </div>

          {/* ── Per-metric score cards ── */}
          <div className="grid grid-cols-2 md:grid-cols-5 gap-3 mb-6">
            {METRIC_NAMES.map(m => {
              const avg  = summary.avgs[m];
              const pass = isMetricPass(m, avg);
              const pct  = summary.scoredCount > 0
                ? ((summary.pass[m] / summary.scoredCount) * 100).toFixed(0)
                : 0;
              const borderColor = pass === null ? 'border-gray-200' : pass ? 'border-emerald-400' : 'border-red-400';
              return (
                <div key={m} className={`bg-white rounded-xl shadow p-4 border-t-4 ${borderColor}`}>
                  <p className="text-xs text-gray-500 mb-1 truncate" title={m}>{m}</p>
                  <p className={`text-2xl font-bold ${pass === null ? 'text-gray-400' : pass ? 'text-emerald-600' : 'text-red-500'}`}>
                    {avg !== null ? avg.toFixed(3) : 'N/A'}
                  </p>
                  <p className="text-xs text-gray-400 mt-0.5">{pct}% pass rate</p>
                  {METRIC_META[m].lowerBetter && (
                    <p className="text-xs text-gray-400 italic">↓ lower = better</p>
                  )}
                </div>
              );
            })}
          </div>

          {/* ── Charts row ── */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
            {/* Average score bar chart */}
            <div className="bg-white rounded-xl shadow p-6 lg:col-span-2">
              <h3 className="text-sm font-semibold text-gray-600 uppercase tracking-wide mb-4">
                Average Score per Metric
              </h3>
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={barData} margin={{ top: 0, right: 10, left: -15, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                  <XAxis dataKey="name" tick={{ fontSize: 11 }} />
                  <YAxis domain={[0, 1]} tick={{ fontSize: 11 }} />
                  <Tooltip formatter={(v, n) => [v.toFixed(3), n]} />
                  <Bar dataKey="score" radius={[4, 4, 0, 0]}>
                    {barData.map((entry, i) => (
                      <Cell key={i} fill={entry.fill} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* Outcome pie */}
            <div className="bg-white rounded-xl shadow p-6">
              <h3 className="text-sm font-semibold text-gray-600 uppercase tracking-wide mb-4">
                Test Case Outcomes
              </h3>
              {outcomeData.length > 0 ? (
                <ResponsiveContainer width="100%" height={220}>
                  <PieChart>
                    <Pie
                      data={outcomeData}
                      cx="50%" cy="50%"
                      outerRadius={75}
                      dataKey="value"
                      label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                      labelLine={false}
                    >
                      {outcomeData.map((entry, i) => (
                        <Cell key={i} fill={entry.fill} />
                      ))}
                    </Pie>
                    <Tooltip />
                  </PieChart>
                </ResponsiveContainer>
              ) : (
                <div className="h-[220px] flex items-center justify-center text-gray-400 text-sm">
                  No data
                </div>
              )}
            </div>
          </div>

          {/* ── Quality score bars ── */}
          <div className="bg-white rounded-xl shadow p-6 mb-6">
            <h3 className="text-sm font-semibold text-gray-600 uppercase tracking-wide mb-1">
              RAG Quality Overview
            </h3>
            <p className="text-xs text-gray-400 mb-5">
              Hallucination bar shows inverted score (higher = less hallucination = better)
            </p>
            <div className="space-y-4">
              {METRIC_NAMES.map(m => {
                const raw = summary.avgs[m];
                const display = raw !== null ? (METRIC_META[m].lowerBetter ? 1 - raw : raw) : null;
                const pct = display !== null ? Math.round(display * 100) : 0;
                const pass = isMetricPass(m, raw);
                const barColor = pass === null ? '#9CA3AF' : pass ? '#10B981' : '#EF4444';
                return (
                  <div key={m}>
                    <div className="flex justify-between items-center mb-1">
                      <span className="text-sm font-medium text-gray-700">{m}</span>
                      <span className={`text-sm font-bold ${pass === null ? 'text-gray-400' : pass ? 'text-emerald-600' : 'text-red-500'}`}>
                        {raw !== null ? `${(raw * 100).toFixed(1)}%` : 'N/A'}
                        {METRIC_META[m].lowerBetter && <span className="text-xs font-normal text-gray-400 ml-1">(↓ lower=better)</span>}
                      </span>
                    </div>
                    <div className="w-full bg-gray-100 rounded-full h-3">
                      <div
                        className="h-3 rounded-full transition-all duration-500"
                        style={{ width: `${pct}%`, backgroundColor: barColor }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* ── Per-question table ── */}
          <div className="bg-white rounded-xl shadow overflow-hidden">
            {/* Table toolbar */}
            <div className="px-6 py-4 border-b border-gray-100 flex flex-wrap items-center gap-3">
              <h3 className="text-sm font-semibold text-gray-700 uppercase tracking-wide mr-auto">
                Per-Question Results ({filteredRows.length} / {currentData.length})
              </h3>
              <input
                type="text"
                placeholder="Search questions…"
                value={searchTerm}
                onChange={e => setSearchTerm(e.target.value)}
                className="border border-gray-200 rounded-lg px-3 py-1.5 text-sm w-48 focus:outline-none focus:ring-2 focus:ring-purple-300"
              />
              <div className="flex gap-1">
                {['all', 'pass', 'fail', 'rejected', 'crash'].map(s => (
                  <button
                    key={s}
                    onClick={() => setStatusFilter(s)}
                    className={`px-3 py-1 text-xs rounded-lg font-medium transition-colors ${
                      statusFilter === s
                        ? 'bg-purple-600 text-white'
                        : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                    }`}
                  >
                    {s.charAt(0).toUpperCase() + s.slice(1)}
                  </button>
                ))}
              </div>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead className="bg-gray-50 text-xs text-gray-500 uppercase tracking-wide">
                  <tr>
                    <th className="px-4 py-3 text-left w-14">ID</th>
                    <th className="px-4 py-3 text-left">Question</th>
                    {METRIC_NAMES.map(m => (
                      <th key={m} className="px-3 py-3 text-center whitespace-nowrap"
                          title={m}>
                        {METRIC_META[m].short}
                      </th>
                    ))}
                    <th className="px-4 py-3 text-center">Status</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredRows.length === 0 && (
                    <tr>
                      <td colSpan={8} className="px-6 py-10 text-center text-gray-400">
                        No results match your filter
                      </td>
                    </tr>
                  )}
                  {filteredRows.map((r, i) => {
                    const isExpanded = expandedRow === (r.id || i);
                    return (
                      <React.Fragment key={r.id || i}>
                        <tr
                          onClick={() => setExpandedRow(isExpanded ? null : (r.id || i))}
                          className={`cursor-pointer border-t border-gray-50 transition-colors ${
                            isExpanded ? 'bg-purple-50' : 'hover:bg-gray-50'
                          }`}
                        >
                          <td className="px-4 py-3 font-mono text-xs text-gray-500">
                            {r.id || `Q${i + 1}`}
                          </td>
                          <td className="px-4 py-3 text-gray-700 max-w-xs truncate" title={r.question}>
                            {r.question}
                          </td>
                          {METRIC_NAMES.map(m => {
                            const s = r.scores?.[m];
                            return (
                              <td key={m} className={`px-3 py-3 text-center tabular-nums ${scoreColorClass(m, s)}`}>
                                {s !== null && s !== undefined ? s.toFixed(3) : '—'}
                              </td>
                            );
                          })}
                          <td className="px-4 py-3 text-center">
                            <StatusBadge r={r} />
                          </td>
                        </tr>
                        {isExpanded && <RowDetail r={r} />}
                      </React.Fragment>
                    );
                  })}
                </tbody>
              </table>
            </div>

            {/* Legend / guide */}
            <div className="px-6 py-4 bg-gray-50 border-t border-gray-100 text-xs text-gray-500 space-y-0.5">
              <p className="font-semibold text-gray-600 mb-1">Metric Guide</p>
              <p>• <strong>AnsRel / Faith</strong>: Primary metrics. ≥ 0.5 to pass. Higher = better.</p>
              <p>• <strong>CtxPre / CtxRec</strong>: ≥ 0.3 to pass. May score 0 for factual/numerical QA — known DeepEval limitation.</p>
              <p>• <strong>Halluc</strong>: ≤ 0.4 to pass. <em>Lower = better</em>. High score = hallucinated claims.</p>
              <p className="mt-1">Click any row to expand question detail and metric reasons.</p>
            </div>
          </div>
        </>
      )}
    </div>
  );
}

// ─── Main Admin Dashboard ─────────────────────────────────────────────────────
const COLORS_PIE = ['#3B82F6', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6'];

function AdminDashboard() {
  // ── Existing state ──
  const [analytics, setAnalytics]           = useState(null);
  const [tickets, setTickets]               = useState([]);
  const [selectedTicket, setSelectedTicket] = useState(null);
  const [loading, setLoading]               = useState(true);
  const [filter, setFilter]                 = useState('all');
  const [timeRange, setTimeRange]           = useState('daily');
  const loadingRef                          = useRef(false);

  // ── Tab state ──
  const [activeTab, setActiveTab]           = useState('analytics'); // 'analytics' | 'evaluation'

  // ── Evaluation state ──
  const [evalLang, setEvalLang]             = useState('english');  // 'english' | 'urdu' | 'roman'
  const [evalStatus, setEvalStatus]         = useState({});         // {english:{exists,run_at,...}, ...}
  const [evalData, setEvalData]             = useState({});         // {english:[...], urdu:[...], roman:[...]}
  const [evalLoading, setEvalLoading]       = useState(false);
  const [evalError, setEvalError]           = useState(null);

  // ── Auth guard ──
  useEffect(() => {
    const token = localStorage.getItem('admin_token');
    if (!token) window.location.hash = '#/admin/login';
  }, []);

  // ── Load analytics + tickets ──
  const loadData = useCallback(async () => {
    if (loadingRef.current) return;
    loadingRef.current = true;
    try {
      const token = localStorage.getItem('admin_token');
      if (!token) { window.location.hash = '#/admin/login'; return; }

      const [analyticsData, ticketsData] = await Promise.all([
        api.getAnalytics(),
        api.adminListTickets(filter === 'all' ? null : filter),
      ]);
      setAnalytics(analyticsData || emptyAnalytics());
      setTickets(ticketsData?.tickets || []);
    } catch (err) {
      if (err.status === 401) {
        localStorage.removeItem('admin_token');
        localStorage.removeItem('admin_id');
        window.location.hash = '#/admin/login';
      } else {
        setAnalytics(emptyAnalytics());
      }
    } finally {
      setLoading(false);
      loadingRef.current = false;
    }
  }, [filter]);

  // ── Load eval status (which languages have reports) ──
  const loadEvalStatus = useCallback(async () => {
    try {
      const status = await api.getEvalStatus();
      setEvalStatus(status);
      return status;
    } catch (err) {
      console.error('Failed to load eval status:', err);
      return {};
    }
  }, []);

  // ── Load a specific language's report ──
  const loadEvalReport = useCallback(async (lang) => {
    // If already loaded, just switch tab
    if (evalData[lang]) {
      setEvalLang(lang);
      setEvalError(null);
      return;
    }
    setEvalLoading(true);
    setEvalError(null);
    setEvalLang(lang);
    try {
      const data = await api.getEvalReportByLanguage(lang);
      setEvalData(prev => ({ ...prev, [lang]: data.results || [] }));
    } catch (err) {
      // 404 = no report run yet for this language — show empty state, don't crash
      if (err?.status === 404) {
        setEvalError(null);   // clear error, empty state will show instead
      } else {
        const detail = err?.data?.detail || err?.message || 'Failed to load report';
        setEvalError(detail);
      }
    } finally {
      setEvalLoading(false);
    }
  }, [evalData]);

  // ── Switch tabs ──
  useEffect(() => {
    if (activeTab === 'analytics') loadData();
    if (activeTab === 'evaluation') {
      // Load status first, then auto-load english only if it exists
      loadEvalStatus().then(status => {
        if (status?.english?.exists && !evalData['english']) {
          loadEvalReport('english');
        }
      });
    }
  }, [activeTab]);                              // eslint-disable-line

  useEffect(() => {
    if (activeTab === 'analytics') loadData();
  }, [loadData]);

  // ── Helpers ──
  function emptyAnalytics() {
    return {
      total_queries: 0, total_rag_answered: 0, total_human_answered: 0,
      total_tickets: 0, active_tickets: 0, in_progress_tickets: 0, resolved_tickets: 0,
      queries_over_time: { daily: [], monthly: [], yearly: [] },
      rag_vs_human: { rag_responses: 0, human_responses: 0, rag_percentage: 0, human_percentage: 0 },
    };
  }

  function formatTime(s) {
    if (!s) return 'N/A';
    const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60);
    return h > 0 ? `${h}h ${m}m` : `${m}m`;
  }

  function handleLogout() {
    localStorage.removeItem('admin_token');
    localStorage.removeItem('admin_id');
    window.location.reload();
  }

  const timeData      = analytics?.queries_over_time?.[timeRange] || [];
  const ragVsHumanData = analytics?.rag_vs_human
    ? [
        { name: 'RAG Answered',   value: analytics.rag_vs_human.rag_responses   || 0 },
        { name: 'Human Answered', value: analytics.rag_vs_human.human_responses  || 0 },
      ]
    : [];

  // ── Loading gate ──
  if (loading && !analytics) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-gray-400">Loading…</div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-100">
      {/* ── Header ── */}
      <header className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <img src={logo} alt="Alkhidmat" className="w-12 h-12" />
              <div>
                <h1 className="text-2xl font-bold text-gray-800">Admin Dashboard</h1>
                {/* Tab switcher */}
                <div className="flex gap-2 mt-1.5">
                  <button
                    onClick={() => setActiveTab('analytics')}
                    className={`px-3 py-1 rounded-lg text-xs font-semibold transition-colors ${
                      activeTab === 'analytics'
                        ? 'bg-blue-600 text-white'
                        : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                    }`}
                  >
                    Analytics & Tickets
                  </button>
                  <button
                    onClick={() => setActiveTab('evaluation')}
                    className={`px-3 py-1 rounded-lg text-xs font-semibold transition-colors ${
                      activeTab === 'evaluation'
                        ? 'bg-purple-600 text-white'
                        : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                    }`}
                  >
                    RAG Evaluation
                  </button>
                </div>
              </div>
            </div>
            <div className="flex gap-3">
              <button
                onClick={() => {
                  localStorage.removeItem('admin_token');
                  localStorage.removeItem('admin_id');
                  window.location.hash = '';
                  window.location.reload();
                }}
                className="px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition-colors text-sm font-medium"
              >
                ← Back
              </button>
              <button
                onClick={handleLogout}
                className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors text-sm font-medium"
              >
                Logout
              </button>
            </div>
          </div>
        </div>
      </header>

      {/* ── Body ── */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">

        {/* ════════════════════ ANALYTICS TAB ════════════════════ */}
        {activeTab === 'analytics' && (
          <>
            {/* Stat cards */}
            {analytics && (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
                <div className="bg-white rounded-lg shadow p-6">
                  <div className="text-sm text-gray-500 mb-1">Total Queries</div>
                  <div className="text-3xl font-bold text-gray-800">{analytics.total_queries || 0}</div>
                </div>
                <div className="bg-white rounded-lg shadow p-6">
                  <div className="text-sm text-gray-500 mb-1">RAG Answered</div>
                  <div className="text-3xl font-bold text-blue-600">{analytics.total_rag_answered || 0}</div>
                  <div className="text-xs text-gray-500 mt-1">{analytics.rag_vs_human?.rag_percentage || 0}% of total</div>
                </div>
                <div className="bg-white rounded-lg shadow p-6">
                  <div className="text-sm text-gray-500 mb-1">Human Answered</div>
                  <div className="text-3xl font-bold text-green-600">{analytics.total_human_answered || 0}</div>
                  <div className="text-xs text-gray-500 mt-1">{analytics.rag_vs_human?.human_percentage || 0}% of total</div>
                </div>
                <div className="bg-white rounded-lg shadow p-6">
                  <div className="text-sm text-gray-500 mb-1">Total Tickets</div>
                  <div className="text-3xl font-bold text-gray-800">{analytics.total_tickets || 0}</div>
                </div>
              </div>
            )}

            {/* Charts */}
            {analytics && (
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
                <div className="bg-white rounded-lg shadow p-6">
                  <div className="flex items-center justify-between mb-4">
                    <h2 className="text-lg font-semibold text-gray-800">Queries Over Time</h2>
                    <div className="flex gap-2">
                      {['daily', 'monthly', 'yearly'].map(r => (
                        <button key={r}
                          onClick={() => setTimeRange(r)}
                          className={`px-3 py-1 text-sm rounded capitalize ${timeRange === r ? 'bg-blue-600 text-white' : 'bg-gray-100 text-gray-700'}`}
                        >{r}</button>
                      ))}
                    </div>
                  </div>
                  {timeData.length > 0 ? (
                    <ResponsiveContainer width="100%" height={300}>
                      <LineChart data={timeData}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey={timeRange === 'daily' ? 'date' : timeRange === 'monthly' ? 'month' : 'year'} tick={{ fontSize: 12 }} />
                        <YAxis tick={{ fontSize: 12 }} />
                        <Tooltip /><Legend />
                        <Line type="monotone" dataKey="total" stroke="#3B82F6" strokeWidth={2} name="Total Queries" />
                        <Line type="monotone" dataKey="rag"   stroke="#10B981" strokeWidth={2} name="RAG Answered"  />
                        <Line type="monotone" dataKey="human" stroke="#F59E0B" strokeWidth={2} name="Human Answered"/>
                      </LineChart>
                    </ResponsiveContainer>
                  ) : (
                    <div className="h-[300px] flex items-center justify-center text-gray-500">No data available</div>
                  )}
                </div>

                <div className="bg-white rounded-lg shadow p-6">
                  <h2 className="text-lg font-semibold text-gray-800 mb-4">RAG vs Human Agent</h2>
                  {ragVsHumanData.some(d => d.value > 0) ? (
                    <ResponsiveContainer width="100%" height={300}>
                      <PieChart>
                        <Pie data={ragVsHumanData} cx="50%" cy="50%" labelLine={false}
                          label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(1)}%`}
                          outerRadius={100} dataKey="value">
                          {ragVsHumanData.map((_, i) => <Cell key={i} fill={COLORS_PIE[i % COLORS_PIE.length]} />)}
                        </Pie>
                        <Tooltip />
                      </PieChart>
                    </ResponsiveContainer>
                  ) : (
                    <div className="h-[300px] flex items-center justify-center text-gray-500">No data available</div>
                  )}
                </div>
              </div>
            )}

            {/* Ticket status cards */}
            {analytics && (
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
                <div className="bg-white rounded-lg shadow p-6">
                  <div className="text-sm text-gray-500 mb-1">Active Tickets</div>
                  <div className="text-3xl font-bold text-yellow-600">{analytics.active_tickets || 0}</div>
                </div>
                <div className="bg-white rounded-lg shadow p-6">
                  <div className="text-sm text-gray-500 mb-1">In Progress</div>
                  <div className="text-3xl font-bold text-blue-600">{analytics.in_progress_tickets || 0}</div>
                </div>
                <div className="bg-white rounded-lg shadow p-6">
                  <div className="text-sm text-gray-500 mb-1">Resolved</div>
                  <div className="text-3xl font-bold text-green-600">{analytics.resolved_tickets || 0}</div>
                </div>
              </div>
            )}

            {/* Performance metrics */}
            {analytics?.average_resolution_time_seconds && (
              <div className="bg-white rounded-lg shadow p-6 mb-8">
                <h2 className="text-lg font-semibold text-gray-800 mb-2">Performance Metrics</h2>
                <div className="flex items-baseline gap-2">
                  <span className="text-2xl font-bold text-gray-800">{formatTime(analytics.average_resolution_time_seconds)}</span>
                  <span className="text-gray-500">Average Resolution Time</span>
                </div>
                {analytics.rag_vs_human?.avg_rag_confidence && (
                  <div className="mt-4">
                    <span className="text-gray-500">Average RAG Confidence: </span>
                    <span className="text-lg font-semibold text-gray-800">
                      {(analytics.rag_vs_human.avg_rag_confidence * 100).toFixed(1)}%
                    </span>
                  </div>
                )}
              </div>
            )}

            {/* Tickets table */}
            <div className="bg-white rounded-lg shadow">
              <div className="p-6 border-b border-gray-200">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-4">
                    <h2 className="text-xl font-semibold text-gray-800">All Tickets</h2>
                    <button onClick={loadData}
                      className="px-3 py-1 text-sm bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 flex items-center gap-1">
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                          d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                      </svg>
                      Refresh
                    </button>
                  </div>
                  <div className="flex gap-2">
                    {[['all','blue'],['active','yellow'],['in_progress','blue'],['resolved','green']].map(([f, c]) => (
                      <button key={f} onClick={() => setFilter(f)}
                        className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                          filter === f ? `bg-${c}-600 text-white` : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                        }`}>
                        {f === 'in_progress' ? 'In Progress' : f.charAt(0).toUpperCase() + f.slice(1)}
                      </button>
                    ))}
                  </div>
                </div>
              </div>

              <div className="divide-y divide-gray-200">
                {tickets.length === 0 ? (
                  <div className="p-8 text-center text-gray-500">No tickets found</div>
                ) : (
                  tickets.map(ticket => (
                    <div key={ticket.ticket_id} onClick={() => setSelectedTicket(ticket)}
                      className={`p-4 cursor-pointer hover:bg-gray-50 transition-colors ${
                        selectedTicket?.ticket_id === ticket.ticket_id ? 'bg-blue-50 border-l-4 border-l-blue-600' : ''
                      }`}>
                      <div className="flex items-start justify-between">
                        <div className="flex-1">
                          <div className="flex items-center gap-2 mb-2">
                            <span className="text-sm font-semibold text-gray-800">#{ticket.ticket_id.slice(0, 8)}</span>
                            <span className={`px-2 py-0.5 text-xs rounded-full ${
                              ticket.status === 'active' ? 'bg-yellow-100 text-yellow-800' :
                              ticket.status === 'in_progress' ? 'bg-blue-100 text-blue-800' :
                              'bg-green-100 text-green-800'}`}>
                              {ticket.status}
                            </span>
                          </div>
                          <p className="text-xs text-gray-500 mb-1">Created: {new Date(ticket.created_at).toLocaleString()}</p>
                          {ticket.resolved_at && (
                            <p className="text-xs text-gray-500">Resolved: {new Date(ticket.resolved_at).toLocaleString()}</p>
                          )}
                          {ticket.response?.content && (
                            <p className="text-sm text-gray-700 mt-2 line-clamp-2">{ticket.response.content}</p>
                          )}
                        </div>
                      </div>
                    </div>
                  ))
                )}
              </div>
            </div>

            {/* Ticket detail modal */}
            {selectedTicket && (
              <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
                <div className="bg-white rounded-lg shadow-xl max-w-2xl w-full max-h-[90vh] overflow-y-auto">
                  <div className="p-6 border-b border-gray-200 flex items-center justify-between">
                    <h3 className="text-xl font-semibold text-gray-800">Ticket #{selectedTicket.ticket_id.slice(0, 8)}</h3>
                    <button onClick={() => setSelectedTicket(null)} className="text-gray-500 hover:text-gray-700">✕</button>
                  </div>
                  <div className="p-6 space-y-4">
                    <div>
                      <label className="text-sm font-medium text-gray-500">Status</label>
                      <div className="mt-1">
                        <span className={`px-3 py-1 text-sm rounded-full ${
                          selectedTicket.status === 'active'      ? 'bg-yellow-100 text-yellow-800' :
                          selectedTicket.status === 'in_progress' ? 'bg-blue-100 text-blue-800' :
                          'bg-green-100 text-green-800'}`}>
                          {selectedTicket.status}
                        </span>
                      </div>
                    </div>
                    <div>
                      <label className="text-sm font-medium text-gray-500">Created At</label>
                      <p className="mt-1 text-gray-800">{new Date(selectedTicket.created_at).toLocaleString()}</p>
                    </div>
                    {selectedTicket.resolved_at && (
                      <div>
                        <label className="text-sm font-medium text-gray-500">Resolved At</label>
                        <p className="mt-1 text-gray-800">{new Date(selectedTicket.resolved_at).toLocaleString()}</p>
                      </div>
                    )}
                    {selectedTicket.agent_id && (
                      <div>
                        <label className="text-sm font-medium text-gray-500">Assigned Agent</label>
                        <p className="mt-1 text-gray-800">{selectedTicket.agent_id}</p>
                      </div>
                    )}
                    {selectedTicket.response?.content && (
                      <div>
                        <label className="text-sm font-medium text-gray-500">Initial Message</label>
                        <p className="mt-1 text-gray-800">{selectedTicket.response.content}</p>
                      </div>
                    )}
                    {selectedTicket.session && (
                      <div>
                        <label className="text-sm font-medium text-gray-500">Session Info</label>
                        <p className="mt-1 text-gray-800">Session ID: {selectedTicket.session.session_id}</p>
                        <p className="text-sm text-gray-500">User ID: {selectedTicket.session.user_id}</p>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            )}
          </>
        )}

        {/* ════════════════════ EVALUATION TAB ════════════════════ */}
        {activeTab === 'evaluation' && (
          <EvalTab
            evalStatus={evalStatus}
            evalLang={evalLang}
            evalData={evalData}
            evalLoading={evalLoading}
            evalError={evalError}
            onSelectLang={loadEvalReport}
          />
        )}
      </div>
    </div>
  );
}

export default AdminDashboard;
