//code by fluric
package ch.ethz.idsc.subare.analysis;

import ch.ethz.idsc.subare.core.MonteCarloInterface;
import ch.ethz.idsc.subare.core.Policy;
import ch.ethz.idsc.subare.core.mc.MonteCarloExploringStarts;
import ch.ethz.idsc.subare.core.td.ExpectedSarsa;
import ch.ethz.idsc.subare.core.td.OriginalSarsa;
import ch.ethz.idsc.subare.core.td.QLearning;
import ch.ethz.idsc.subare.core.td.Sarsa;
import ch.ethz.idsc.subare.core.td.TrueOnlineSarsa;
import ch.ethz.idsc.subare.core.util.ConstantLearningRate;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.EGreedyPolicy;
import ch.ethz.idsc.subare.core.util.ExactFeatureMapper;
import ch.ethz.idsc.subare.core.util.ExploringStarts;
import ch.ethz.idsc.subare.core.util.FeatureMapper;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;

public enum AnalysisAlgorithms {
  ORIGINALSARSA() {
    @Override
    public AnalysisAlgorithm supply() {
      return new AnalysisAlgorithm() {
        @Override
        public Tensor analyse(MonteCarloInterface mcInterface, int batches, DiscreteQsa optimalQsa) {
          Tensor XYsarsa = Tensors.empty();
          DiscreteQsa qsaSarsa = DiscreteQsa.build(mcInterface);
          final Sarsa sarsa = new OriginalSarsa(mcInterface, qsaSarsa, ConstantLearningRate.of(RealScalar.of(0.05)));
          sarsa.setExplore(RealScalar.of(.1));
          long start = System.currentTimeMillis();
          for (int index = 0; index < batches; ++index) {
            // System.out.println("starting batch " + (index + 1) + " of " + batches);
            Policy policy = EGreedyPolicy.bestEquiprobable(mcInterface, sarsa.qsa(), RealScalar.of(.1));
            ExploringStarts.batch(mcInterface, policy, 1, sarsa);
            XYsarsa.append(Tensors.vector(RealScalar.of(index).number(), AnalysisUtils.getLinearQsaError(sarsa.qsa(), optimalQsa).number()));
          }
          System.out.println("Time for OriginalSarsa: " + (System.currentTimeMillis() - start) / 1000.0 + "s");
          System.out.println("Error of OriginalSarsa: Linear: " + AnalysisUtils.getLinearQsaError(sarsa.qsa(), optimalQsa).number().doubleValue()
              + " Quadratic: " + AnalysisUtils.getSquareQsaError(sarsa.qsa(), optimalQsa).number().doubleValue());
          // Policies.print(GreedyPolicy.bestEquiprobable(airport, sarsa.qsa()), airport.states());
          return XYsarsa;
        }

        @Override
        public String getName() {
          return "OriginalSarsa";
        }
      };
    }
  }, //
  EXPECTEDSARSA() {
    @Override
    public AnalysisAlgorithm supply() {
      return new AnalysisAlgorithm() {
        @Override
        public Tensor analyse(MonteCarloInterface mcInterface, int batches, DiscreteQsa optimalQsa) {
          Tensor XYsarsa = Tensors.empty();
          DiscreteQsa qsaSarsa = DiscreteQsa.build(mcInterface);
          final Sarsa sarsa = new ExpectedSarsa(mcInterface, qsaSarsa, ConstantLearningRate.of(RealScalar.of(0.05)));
          sarsa.setExplore(RealScalar.of(.1));
          long start = System.currentTimeMillis();
          for (int index = 0; index < batches; ++index) {
            // System.out.println("starting batch " + (index + 1) + " of " + batches);
            Policy policy = EGreedyPolicy.bestEquiprobable(mcInterface, sarsa.qsa(), RealScalar.of(.1));
            ExploringStarts.batch(mcInterface, policy, 1, sarsa);
            XYsarsa.append(Tensors.vector(RealScalar.of(index).number(), AnalysisUtils.getLinearQsaError(sarsa.qsa(), optimalQsa).number()));
          }
          System.out.println("Time for ExpectedSarsa: " + (System.currentTimeMillis() - start) / 1000.0 + "s");
          System.out.println("Error of ExpectedSarsa: Linear: " + AnalysisUtils.getLinearQsaError(sarsa.qsa(), optimalQsa).number().doubleValue()
              + " Quadratic: " + AnalysisUtils.getSquareQsaError(sarsa.qsa(), optimalQsa).number().doubleValue());
          // Policies.print(GreedyPolicy.bestEquiprobable(airport, sarsa.qsa()), airport.states());
          return XYsarsa;
        }

        @Override
        public String getName() {
          return "ExpectedSarsa";
        }
      };
    }
  }, //
  QLEARNINGSARSA() {
    @Override
    public AnalysisAlgorithm supply() throws Exception {
      return new AnalysisAlgorithm() {
        @Override
        public Tensor analyse(MonteCarloInterface mcInterface, int batches, DiscreteQsa optimalQsa) {
          Tensor XYsarsa = Tensors.empty();
          DiscreteQsa qsaSarsa = DiscreteQsa.build(mcInterface);
          final Sarsa sarsa = new QLearning(mcInterface, qsaSarsa, ConstantLearningRate.of(RealScalar.of(0.05)));
          sarsa.setExplore(RealScalar.of(.1));
          long start = System.currentTimeMillis();
          for (int index = 0; index < batches; ++index) {
            // System.out.println("starting batch " + (index + 1) + " of " + batches);
            Policy policy = EGreedyPolicy.bestEquiprobable(mcInterface, sarsa.qsa(), RealScalar.of(.1));
            ExploringStarts.batch(mcInterface, policy, 1, sarsa);
            XYsarsa.append(Tensors.vector(RealScalar.of(index).number(), AnalysisUtils.getLinearQsaError(sarsa.qsa(), optimalQsa).number()));
          }
          System.out.println("Time for QLearningSarsa: " + (System.currentTimeMillis() - start) / 1000.0 + "s");
          System.out.println("Error of QLearningSarsa: Linear: " + AnalysisUtils.getLinearQsaError(sarsa.qsa(), optimalQsa).number().doubleValue()
              + " Quadratic: " + AnalysisUtils.getSquareQsaError(sarsa.qsa(), optimalQsa).number().doubleValue());
          // Policies.print(GreedyPolicy.bestEquiprobable(airport, sarsa.qsa()), airport.states());
          return XYsarsa;
        }

        @Override
        public String getName() {
          return "QLearningSarsa";
        }
      };
    }
  }, //
  MONTECARLO() {
    @Override
    public AnalysisAlgorithm supply() throws Exception {
      return new AnalysisAlgorithm() {
        @Override
        public Tensor analyse(MonteCarloInterface mcInterface, int batches, DiscreteQsa optimalQsa) {
          Tensor XYmc = Tensors.empty();
          MonteCarloExploringStarts mces = new MonteCarloExploringStarts(mcInterface);
          long start = System.currentTimeMillis();
          for (int index = 0; index < batches; ++index) {
            // System.out.println("starting batch " + (index + 1) + " of " + batches);
            Policy policyMC = EGreedyPolicy.bestEquiprobable(mcInterface, mces.qsa(), RealScalar.of(.1));
            ExploringStarts.batch(mcInterface, policyMC, mces);
            XYmc.append(Tensors.vector(RealScalar.of(index).number(), AnalysisUtils.getLinearQsaError(mces.qsa(), optimalQsa).number()));
          }
          System.out.println("Time for MonteCarlo: " + (System.currentTimeMillis() - start) / 1000.0 + "s");
          System.out.println("Error of MonteCarlo: Linear:" + AnalysisUtils.getLinearQsaError(mces.qsa(), optimalQsa).number().doubleValue() + " Quadratic: "
              + AnalysisUtils.getSquareQsaError(mces.qsa(), optimalQsa).number().doubleValue());
          // Policies.print(GreedyPolicy.bestEquiprobable(airport, mces.qsa()), airport.states());
          return XYmc;
        }

        @Override
        public String getName() {
          return "MonteCarlo";
        }
      };
    }
  }, //
  TRUEONLINESARSA() {
    @Override
    public AnalysisAlgorithm supply() throws Exception {
      return new AnalysisAlgorithm() {
        @Override
        public Tensor analyse(MonteCarloInterface mcInterface, int batches, DiscreteQsa optimalQsa) {
          Tensor XYtoSarsa = Tensors.empty();
          FeatureMapper mapper = new ExactFeatureMapper(mcInterface);
          TrueOnlineSarsa toSarsa = new TrueOnlineSarsa(mcInterface, RealScalar.of(0.7), RealScalar.of(0.2), RealScalar.of(1), mapper);
          long start = System.currentTimeMillis();
          for (int index = 0; index < batches; ++index) {
            // System.out.println("starting batch " + (index + 1) + " of " + batches);
            toSarsa.executeEpisode(RealScalar.of(0.1));
            DiscreteQsa toQsa = toSarsa.getQsa();
            XYtoSarsa.append(Tensors.vector(RealScalar.of(index).number(), AnalysisUtils.getLinearQsaError(toQsa, optimalQsa).number()));
          }
          System.out.println("Time for TrueOnlineSarsa: " + (System.currentTimeMillis() - start) / 1000.0 + "s");
          DiscreteQsa toQsa = toSarsa.getQsa();
          // System.out.println(toSarsa.getW());
          // toSarsa.printValues();
          // toSarsa.printPolicy();
          System.out.println("Error of TrueOnlineSarsa: Linear: " + AnalysisUtils.getLinearQsaError(toQsa, optimalQsa).number().doubleValue() + " Quadratic: "
              + AnalysisUtils.getSquareQsaError(toQsa, optimalQsa).number().doubleValue());
          return XYtoSarsa;
        }

        @Override
        public String getName() {
          return "TrueOnlineSarsa";
        }
      };
    }
  },//
  ;
  // ---
  public abstract AnalysisAlgorithm supply() throws Exception;
}