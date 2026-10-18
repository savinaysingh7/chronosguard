import React, { useState } from 'react';
import {
  GraduationCap,
  Car,
  Swords,
  Stethoscope,
  HelpCircle,
  Wrench,
  Sliders,
  ShieldCheck,
  AlertTriangle
} from 'lucide-react';

const METAPHORS = [
  {
    icon: Car,
    title: '1. The Self-Driving Car in a Blizzard',
    concept: 'Data Drift (Covariate Shift: P(X) changes)',
    badge: 'Covariate Shift',
    badgeColor: 'bg-blue-500/20 text-blue-300 border-blue-500/40',
    story:
      'Imagine you train a Tesla self-driving car in sunny California. It drives with 99.9% accuracy. Then winter comes and you ship it to Boston during a blizzard. The cameras still work, the computer didn’t crash, but the car starts swerving dangerously.',
    takeaway:
      'The world changed beneath its wheels. In loan risk, this happens when surprise inflation suddenly doubles applicants’ debt-to-income ratios.'
  },
  {
    icon: Swords,
    title: '2. The Changing Rules of Chess',
    concept: 'Concept Drift (Structural Shift: P(Y|X) flips)',
    badge: 'Concept Drift',
    badgeColor: 'bg-rose-500/20 text-rose-300 border-rose-500/40',
    story:
      'You have practiced chess for 10 years and know every opening move. But halfway through the championship match, your opponent announces that bishops can now jump like knights. The pieces look the same, the board is identical, but the fundamental definition of a winning move has flipped.',
    takeaway:
      'Even if borrower incomes look identical on paper, macro shocks cause people to default who never would have defaulted historically.'
  },
  {
    icon: Stethoscope,
    title: '3. The ER Doctor Without Lab Tests',
    concept: 'Label-Free Monitoring (CBPE)',
    badge: 'Label-Free CBPE',
    badgeColor: 'bg-cyan-500/20 text-cyan-300 border-cyan-500/40',
    story:
      'When an emergency patient arrives in the ICU, blood lab results take 24 to 48 hours to arrive. Doctors cannot wait for the lab report (the true outcome Y is missing). Instead, they look at vital signs—pulse, blood pressure, oxygen—to predict patient deterioration right now.',
    takeaway:
      'ChronosGuard does the exact same thing: it accurately estimates model accuracy without waiting 90 days for loan default outcomes.'
  },
  {
    icon: HelpCircle,
    title: '4. The "I Don’t Know" Button',
    concept: 'Conformal Prediction Sets',
    badge: 'Conformal Sets',
    badgeColor: 'bg-purple-500/20 text-purple-300 border-purple-500/40',
    story:
      'Most AI models behave like an overconfident student: even when they are totally guessing, they output a single answer with fake certainty. Conformal Prediction gives the model an honest "I don’t know" button.',
    takeaway:
      'When conditions are normal, it says {SAFE}. When unfamiliar data arrives, its answer widens to {SAFE, DEFAULT}. That widening gives mathematical proof of decay!'
  },
  {
    icon: Wrench,
    title: '5. The Mechanic vs Total Engine Replacement',
    concept: 'Cost-Aware Retraining ROI',
    badge: 'Cost Optimization',
    badgeColor: 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40',
    story:
      'You don’t buy a brand-new $8,000 engine just because an oil cap is loose. Retraining ML models burns cloud compute, engineering hours, and triggers regulatory reviews.',
    takeaway:
      'ChronosGuard only triggers automated retraining when the dollar loss from bad predictions exceeds the compute cost of retraining.'
  }
];

const GLOSSARY = [
  { term: 'Covariate Shift (Data Drift)', math: 'P(X_train) ≠ P(X_test)', meaning: 'The environment changed, but the rules stayed the same (e.g. inflation went up).' },
  { term: 'Concept Drift', math: 'P(Y|X_train) ≠ P(Y|X_test)', meaning: 'The underlying rules of human behavior flipped (e.g. fraudsters invented a totally new scheme).' },
  { term: 'Model Decay', math: 'Metric(T_t) < Metric(T_0)', meaning: 'The model gets progressively dumber because it is living in the past.' },
  { term: 'Population Stability Index (PSI)', math: 'Σ (Actual% - Expected%) × ln(A%/E%)', meaning: 'A statistical smoke alarm that rings when today’s customer distribution looks too different from yesterday.' },
  { term: 'Maximum Mean Discrepancy (MMD)', math: '||μ_P - μ_Q||_H', meaning: 'A high-dimensional sensor that catches subtle multi-variable shifts that 1D tests miss.' },
  { term: 'CBPE Estimation', math: 'E[ConfMatrix | P(Y|X)]', meaning: 'Predicting your exam grade without having the answer key, by looking at your confidence.' },
  { term: 'Conformal Prediction', math: 'P(Y ∈ C(X)) ≥ 1 - α', meaning: 'A mathematical guarantee that widens the answer to include multiple possibilities when uncertain.' },
  { term: 'SHAP Attribution', math: 'φ_i(v)', meaning: 'A root-cause detective that tells you exactly which feature is responsible for pushing the score into danger.' }
];

export default function Eli5View() {
  const [volatility, setVolatility] = useState(0);

  // Dynamic calculations for simulator
  const simulatedAcc = Math.max(45.0, 88.5 - volatility * 0.42).toFixed(1);
  const simulatedBleed = Math.round(25000 + Math.pow(volatility, 1.8) * 85);
  const conformalWidth = (1.05 + volatility * 0.008).toFixed(2);
  const isRetrainTriggered = volatility >= 55;

  return (
    <div className="p-8 space-y-8 max-w-7xl mx-auto">
      {/* Title */}
      <div>
        <h3 className="text-xl font-extrabold text-white tracking-tight flex items-center gap-2">
          <GraduationCap className="w-6 h-6 text-cyan-400" />
          The Beginner's Guide: Why ML Models Silently Break in the Real World
        </h3>
        <p className="text-xs text-slate-400 mt-1 max-w-3xl leading-relaxed">
          If you are new to machine learning, statistics, or MLOps, welcome! Here is the fundamental truth:{' '}
          <span className="text-white font-semibold">deploying a model is only 10% of the work</span>. The other 90% is making sure it doesn’t silently self-destruct as the world shifts around it.
        </p>
      </div>

      {/* Interactive Shift-the-World Sandbox */}
      <div className="p-6 rounded-2xl bg-gradient-to-r from-slate-900 via-slate-900 to-cyan-950/40 border border-slate-800 space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div>
            <span className="text-xs font-bold text-cyan-400 uppercase tracking-wider flex items-center gap-1.5 mb-1">
              <Sliders className="w-4 h-4" />
              Interactive Story Simulator
            </span>
            <h4 className="text-base font-bold text-white tracking-tight">
              "Shift the World" Economic Volatility Simulator
            </h4>
            <p className="text-xs text-slate-400 mt-0.5">
              Drag the slider to introduce economic stress and watch how an unmonitored loan model decays
            </p>
          </div>

          <div className="text-right">
            <span className="text-xs text-slate-400 block font-medium">Macro Stress Level:</span>
            <span className="text-2xl font-black text-cyan-400 font-mono">{volatility}%</span>
          </div>
        </div>

        {/* Slider */}
        <input
          type="range"
          min={0}
          max={100}
          step={5}
          value={volatility}
          onChange={(e) => setVolatility(Number(e.target.value))}
          className="w-full accent-cyan-500 cursor-pointer h-2 bg-slate-950 rounded-lg"
        />

        {/* Dynamic Simulator KPIs */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 pt-2 font-mono">
          <div className="p-4 rounded-xl bg-slate-950/80 border border-slate-800">
            <span className="text-[11px] text-slate-400 block">Model True Accuracy</span>
            <span className="text-2xl font-bold text-white">{simulatedAcc}%</span>
            <span className="text-[10px] text-slate-400 block mt-0.5">
              {volatility === 0 ? 'Peak Californian Sunny Day' : `Dropped by ${(88.5 - simulatedAcc).toFixed(1)}%`}
            </span>
          </div>

          <div className="p-4 rounded-xl bg-slate-950/80 border border-slate-800">
            <span className="text-[11px] text-slate-400 block">Silent Financial Loss</span>
            <span className="text-2xl font-bold text-rose-400">${simulatedBleed.toLocaleString()}</span>
            <span className="text-[10px] text-slate-400 block mt-0.5">Unhedged default approvals</span>
          </div>

          <div className="p-4 rounded-xl bg-slate-950/80 border border-slate-800">
            <span className="text-[11px] text-slate-400 block">Conformal Uncertainty</span>
            <span className="text-2xl font-bold text-purple-400">{conformalWidth} sets</span>
            <span className="text-[10px] text-slate-400 block mt-0.5">Widened from confident {`{0}`}</span>
          </div>

          <div className="p-4 rounded-xl bg-slate-950/80 border border-slate-800">
            <span className="text-[11px] text-slate-400 block">ChronosGuard Radar</span>
            <span
              className={`text-sm font-bold block mt-1.5 ${
                isRetrainTriggered ? 'text-rose-400 animate-pulse' : volatility >= 25 ? 'text-amber-400' : 'text-emerald-400'
              }`}
            >
              {isRetrainTriggered ? '🔴 RETRAIN TRIGGERED' : volatility >= 25 ? '🟡 WARNING ELEVATED' : '🟢 SYSTEM STABLE'}
            </span>
          </div>
        </div>

        {/* Dynamic Explanation Alert */}
        <div
          className={`p-4 rounded-xl border text-xs flex items-center gap-3 ${
            isRetrainTriggered
              ? 'bg-rose-500/10 border-rose-500/30 text-rose-200'
              : volatility >= 25
              ? 'bg-amber-500/10 border-amber-500/30 text-amber-200'
              : 'bg-emerald-500/10 border-emerald-500/30 text-emerald-200'
          }`}
        >
          {isRetrainTriggered ? (
            <AlertTriangle className="w-5 h-5 text-rose-400 shrink-0" />
          ) : (
            <ShieldCheck className="w-5 h-5 text-emerald-400 shrink-0" />
          )}
          <span>
            {isRetrainTriggered ? (
              <b>Autonomous Retraining Activated:</b>
            ) : volatility >= 25 ? (
              <b>Elevated Uncertainty:</b>
            ) : (
              <b>Model In-Distribution:</b>
            )}{' '}
            {isRetrainTriggered
              ? 'Financial decay loss has exceeded the compute retraining expense. ChronosGuard autonomously hot-swapped the model on a sliding window without human downtime.'
              : volatility >= 25
              ? 'Input distributions are shifting away from training. Conformal set sizes are widening, warning of impending decay.'
              : 'Inputs match historical training. High confidence, zero unnecessary retraining churn.'}
          </span>
        </div>
      </div>

      {/* The 5 Story Cards */}
      <div className="space-y-4">
        <h4 className="text-sm font-bold text-white tracking-tight uppercase">
          5 Everyday Stories That Explain Enterprise MLOps
        </h4>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {METAPHORS.map((m, i) => {
            const Icon = m.icon;
            return (
              <div
                key={i}
                className="p-6 rounded-2xl bg-slate-900/60 border border-slate-800 space-y-3 flex flex-col justify-between"
              >
                <div>
                  <div className="flex items-center justify-between mb-3">
                    <div className="w-10 h-10 rounded-xl bg-slate-800 flex items-center justify-center text-cyan-400">
                      <Icon className="w-5 h-5" />
                    </div>
                    <span className={`text-[10px] font-mono font-bold px-2 py-0.5 rounded border ${m.badgeColor}`}>
                      {m.badge}
                    </span>
                  </div>

                  <h5 className="text-sm font-bold text-white tracking-tight">{m.title}</h5>
                  <p className="text-xs text-slate-300 mt-2 leading-relaxed">{m.story}</p>
                </div>

                <div className="pt-3 border-t border-slate-800/80">
                  <p className="text-xs text-cyan-300 font-medium">
                    <b>Why it matters:</b> {m.takeaway}
                  </p>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Jargon Buster Table */}
      <div className="p-6 rounded-2xl bg-slate-900/60 border border-slate-800 space-y-3">
        <h4 className="text-sm font-bold text-white tracking-tight uppercase">
          📚 Jargon Buster Glossary: Academic Terms Translated to Plain English
        </h4>

        <div className="overflow-x-auto rounded-xl border border-slate-800/80">
          <table className="w-full text-left text-xs">
            <thead className="bg-slate-950 text-slate-400 font-semibold border-b border-slate-800">
              <tr>
                <th className="py-2.5 px-4">Academic Term</th>
                <th className="py-2.5 px-4">Mathematical Notation</th>
                <th className="py-2.5 px-4">Plain English Meaning</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/50 font-mono text-slate-300">
              {GLOSSARY.map((g, i) => (
                <tr key={i} className="hover:bg-slate-800/30">
                  <td className="py-2 px-4 font-bold text-white">{g.term}</td>
                  <td className="py-2 px-4 text-cyan-400">{g.math}</td>
                  <td className="py-2 px-4 font-sans text-slate-300">{g.meaning}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
