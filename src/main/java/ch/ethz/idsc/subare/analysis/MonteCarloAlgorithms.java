// code by fluric
package ch.ethz.idsc.subare.analysis;

import ch.ethz.idsc.subare.core.LearningRate;
import ch.ethz.idsc.subare.core.MonteCarloInterface;
import ch.ethz.idsc.subare.core.Policy;
import ch.ethz.idsc.subare.core.mc.MonteCarloExploringStarts;
import ch.ethz.idsc.subare.core.td.DoubleSarsa;
import ch.ethz.idsc.subare.core.td.ExpectedSarsa;
import ch.ethz.idsc.subare.core.td.OriginalSarsa;
import ch.ethz.idsc.subare.core.td.QLearning;
import ch.ethz.idsc.subare.core.td.Sarsa;
import ch.ethz.idsc.subare.core.td.SarsaType;
import ch.ethz.idsc.subare.core.td.TrueOnlineSarsa;
import ch.ethz.idsc.subare.core.util.ConstantLearningRate;
import ch.ethz.idsc.subare.core.util.DefaultLearningRate;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.DiscreteValueFunctions;
import ch.ethz.idsc.subare.core.util.EGreedyPolicy;
import ch.ethz.idsc.subare.core.util.ExactFeatureMapper;
import ch.ethz.idsc.subare.core.util.ExploringStarts;
import ch.ethz.idsc.subare.core.util.FeatureMapper;
import ch.ethz.idsc.subare.util.Stopwatch;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;

public enum MonteCarloAlgorithms {
  OriginalSarsa() {
    @Override
    public Tensor analyse(MonteCarloInterface monteCarloInterface, int batches, DiscreteQsa optimalQsa, MonteCarloErrorAnalysis errorAnalysis) {
      Tensor XYsarsa = Tensors.empty();
      DiscreteQsa qsaSarsa = DiscreteQsa.build(monteCarloInterface);
      final Sarsa sarsa = new OriginalSarsa(monteCarloInterface, qsaSarsa, ConstantLearningRate.of(RealScalar.of(0.05)));
      sarsa.setExplore(RealScalar.of(.1));
      Stopwatch stopwatch = Stopwatch.started();
      for (int index = 0; index < batches; ++index) {
        // System.out.println("starting batch " + (index + 1) + " of " + batches);
        Policy policy = EGreedyPolicy.bestEquiprobable(monteCarloInterface, sarsa.qsa(), RealScalar.of(.1));
        ExploringStarts.batch(monteCarloInterface, policy, 1, sarsa);
        XYsarsa.append(Tensors.vector(RealScalar.of(index).number(), errorAnalysis.getError(monteCarloInterface, optimalQsa, sarsa.qsa()).number()));
      }
      System.out.println("Time for OriginalSarsa: " + stopwatch.display_seconds() + "s");
      // Policies.print(GreedyPolicy.bestEquiprobable(monteCarloInterface, sarsa.qsa()), monteCarloInterface.states());
      return XYsarsa;
    }
  }, //
  ExpectedSarsa() {
    @Override
    public Tensor analyse(MonteCarloInterface monteCarloInterface, int batches, DiscreteQsa optimalQsa, MonteCarloErrorAnalysis errorAnalysis) {
      Tensor XYsarsa = Tensors.empty();
      DiscreteQsa qsaSarsa = DiscreteQsa.build(monteCarloInterface);
      final Sarsa sarsa = new ExpectedSarsa(monteCarloInterface, qsaSarsa, ConstantLearningRate.of(RealScalar.of(0.05)));
      sarsa.setExplore(RealScalar.of(.1));
      Stopwatch stopwatch = Stopwatch.started();
      for (int index = 0; index < batches; ++index) {
        // System.out.println("starting batch " + (index + 1) + " of " + batches);
        Policy policy = EGreedyPolicy.bestEquiprobable(monteCarloInterface, sarsa.qsa(), RealScalar.of(.1));
        ExploringStarts.batch(monteCarloInterface, policy, 1, sarsa);
        XYsarsa.append(Tensors.vector(RealScalar.of(index).number(), errorAnalysis.getError(monteCarloInterface, optimalQsa, sarsa.qsa()).number()));
      }
      System.out.println("Time for ExpectedSarsa: " + stopwatch.display_seconds() + "s");
      // Policies.print(GreedyPolicy.bestEquiprobable(monteCarloInterface, sarsa.qsa()), monteCarloInterface.states());
      return XYsarsa;
    }
  }, //
  QLearningSarsa() {
    @Override
    public Tensor analyse(MonteCarloInterface monteCarloInterface, int batches, DiscreteQsa optimalQsa, MonteCarloErrorAnalysis errorAnalysis) {
      Tensor XYsarsa = Tensors.empty();
      DiscreteQsa qsaSarsa = DiscreteQsa.build(monteCarloInterface);
      final Sarsa sarsa = new QLearning(monteCarloInterface, qsaSarsa, ConstantLearningRate.of(RealScalar.of(0.05)));
      sarsa.setExplore(RealScalar.of(.1));
      Stopwatch stopwatch = Stopwatch.started();
      for (int index = 0; index < batches; ++index) {
        // System.out.println("starting batch " + (index + 1) + " of " + batches);
        Policy policy = EGreedyPolicy.bestEquiprobable(monteCarloInterface, sarsa.qsa(), RealScalar.of(.1));
        ExploringStarts.batch(monteCarloInterface, policy, 1, sarsa);
        XYsarsa.append(Tensors.vector(RealScalar.of(index).number(), errorAnalysis.getError(monteCarloInterface, optimalQsa, sarsa.qsa()).number()));
      }
      System.out.println("Time for QLearningSarsa: " + stopwatch.display_seconds() + "s");
      // Policies.print(GreedyPolicy.bestEquiprobable(monteCarloInterface, sarsa.qsa()), monteCarloInterface.states());
      return XYsarsa;
    }
  }, //
  DoubleQLearningSarsa() {
    @Override
    public Tensor analyse(MonteCarloInterface monteCarloInterface, int batches, DiscreteQsa optimalQsa, MonteCarloErrorAnalysis errorAnalysis) {
      Tensor XYsarsa = Tensors.empty();
      DiscreteQsa qsa1 = DiscreteQsa.build(monteCarloInterface);
      DiscreteQsa qsa2 = DiscreteQsa.build(monteCarloInterface);
      DoubleSarsa sarsa = new DoubleSarsa(SarsaType.QLEARNING, monteCarloInterface, //
          qsa1, qsa2, //
          DefaultLearningRate.of(1, .51), //
          DefaultLearningRate.of(1, .51));
      sarsa.setExplore(RealScalar.of(.1));
      Stopwatch stopwatch = Stopwatch.started();
      for (int index = 0; index < batches; ++index) {
        // System.out.println("starting batch " + (index + 1) + " of " + batches);
        Policy policy = sarsa.getEGreedy();
        ExploringStarts.batch(monteCarloInterface, policy, 1, sarsa);
        XYsarsa.append(Tensors.vector(RealScalar.of(index).number(),
            errorAnalysis.getError(monteCarloInterface, optimalQsa, DiscreteValueFunctions.average(qsa1, qsa2)).number()));
      }
      System.out.println("Time for DoubleQLearningSarsa: " + stopwatch.display_seconds() + "s");
      // Policies.print(sarsa.getGreedy(), monteCarloInterface.states());
      return XYsarsa;
    }
  }, //
  MonteCarlo() {
    @Override
    public Tensor analyse(MonteCarloInterface monteCarloInterface, int batches, DiscreteQsa optimalQsa, MonteCarloErrorAnalysis errorAnalysis) {
      Tensor XYmc = Tensors.empty();
      MonteCarloExploringStarts mces = new MonteCarloExploringStarts(monteCarloInterface);
      Stopwatch stopwatch = Stopwatch.started();
      for (int index = 0; index < batches; ++index) {
        // System.out.println("starting batch " + (index + 1) + " of " + batches);
        Policy policyMC = EGreedyPolicy.bestEquiprobable(monteCarloInterface, mces.qsa(), RealScalar.of(.1));
        ExploringStarts.batch(monteCarloInterface, policyMC, mces);
        XYmc.append(Tensors.vector(RealScalar.of(index).number(), errorAnalysis.getError(monteCarloInterface, optimalQsa, mces.qsa()).number()));
      }
      System.out.println("Time for MonteCarlo: " + stopwatch.display_seconds() + "s");
      // Policies.print(GreedyPolicy.bestEquiprobable(monteCarloInterface, mces.qsa()), monteCarloInterface.states());
      return XYmc;
    }
  }, //
  TrueOnlineSarsa() {
    @Override
    public Tensor analyse(MonteCarloInterface monteCarloInterface, int batches, DiscreteQsa optimalQsa, MonteCarloErrorAnalysis errorAnalysis) {
      Tensor XYtoSarsa = Tensors.empty();
      LearningRate learningRate = ConstantLearningRate.of(RealScalar.of(0.2), false);
      FeatureMapper mapper = new ExactFeatureMapper(monteCarloInterface);
      TrueOnlineSarsa toSarsa = new TrueOnlineSarsa(monteCarloInterface, RealScalar.of(0.7), learningRate, mapper);
      Stopwatch stopwatch = Stopwatch.started();
      for (int index = 0; index < batches; ++index) {
        // System.out.println("starting batch " + (index + 1) + " of " + batches);
        toSarsa.executeBatch(RealScalar.of(0.1));
        DiscreteQsa toQsa = toSarsa.getQsa();
        XYtoSarsa.append(Tensors.vector(RealScalar.of(index).number(), errorAnalysis.getError(monteCarloInterface, optimalQsa, toQsa).number()));
      }
      System.out.println("Time for TrueOnlineSarsa: " + stopwatch.display_seconds() + "s");
      DiscreteQsa toQsa = toSarsa.getQsa();
      // System.out.println(toSarsa.getW());
      // toSarsa.printValues();
      // toSarsa.printPolicy();
      return XYtoSarsa;
    }
  }, //
  TrueOnlineSarsaWarmStart() {
    @Override
    public Tensor analyse(MonteCarloInterface monteCarloInterface, int batches, DiscreteQsa optimalQsa, MonteCarloErrorAnalysis errorAnalysis) {
      Tensor XYtoSarsa = Tensors.empty();
      LearningRate learningRate = ConstantLearningRate.of(RealScalar.of(0.2), true);
      FeatureMapper mapper = new ExactFeatureMapper(monteCarloInterface);
      TrueOnlineSarsa toSarsa = new TrueOnlineSarsa(monteCarloInterface, RealScalar.of(0.7), learningRate, mapper);
      Stopwatch stopwatch = Stopwatch.started();
      for (int index = 0; index < batches; ++index) {
        // System.out.println("starting batch " + (index + 1) + " of " + batches);
        toSarsa.executeBatch(RealScalar.of(0.1));
        DiscreteQsa toQsa = toSarsa.getQsa();
        XYtoSarsa.append(Tensors.vector(RealScalar.of(index).number(), errorAnalysis.getError(monteCarloInterface, optimalQsa, toQsa).number()));
      }
      System.out.println("Time for TrueOnlineSarsaWarmStart: " + stopwatch.display_seconds() + "s");
      DiscreteQsa toQsa = toSarsa.getQsa();
      // System.out.println(toSarsa.getW());
      // toSarsa.printValues();
      // toSarsa.printPolicy();
      return XYtoSarsa;
    }
  }, //
  TrueOnlineSarsaZero() {
    @Override
    public Tensor analyse(MonteCarloInterface monteCarloInterface, int batches, DiscreteQsa optimalQsa, MonteCarloErrorAnalysis errorAnalysis) {
      Tensor XYtoSarsa = Tensors.empty();
      FeatureMapper mapper = new ExactFeatureMapper(monteCarloInterface);
      TrueOnlineSarsa toSarsa = new TrueOnlineSarsa(monteCarloInterface, RealScalar.of(0.0), ConstantLearningRate.of(RealScalar.of(0.05)), mapper);
      Stopwatch stopwatch = Stopwatch.started();
      for (int index = 0; index < batches; ++index) {
        // System.out.println("starting batch " + (index + 1) + " of " + batches);
        toSarsa.executeBatch(RealScalar.of(0.1));
        DiscreteQsa toQsa = toSarsa.getQsa();
        XYtoSarsa.append(Tensors.vector(RealScalar.of(index).number(), errorAnalysis.getError(monteCarloInterface, optimalQsa, toQsa).number()));
      }
      System.out.println("Time for TrueOnlineSarsaZero: " + stopwatch.display_seconds() + "s");
      DiscreteQsa toQsa = toSarsa.getQsa();
      // System.out.println(toSarsa.getW());
      // toSarsa.printValues();
      // toSarsa.printPolicy();
      return XYtoSarsa;
    }
  }, //
  TrueOnlineSarsaTest() {
    @Override
    public Tensor analyse(MonteCarloInterface monteCarloInterface, int batches, DiscreteQsa optimalQsa, MonteCarloErrorAnalysis errorAnalysis) {
      Tensor XYtoSarsa = Tensors.empty();
      LearningRate learningRate = ConstantLearningRate.of(RealScalar.of(0.05));
      FeatureMapper mapper = new ExactFeatureMapper(monteCarloInterface);
      TrueOnlineSarsa toSarsa = new TrueOnlineSarsa(monteCarloInterface, RealScalar.of(0.7), learningRate, mapper);
      Stopwatch stopwatch = Stopwatch.started();
      for (int index = 0; index < batches; ++index) {
        // System.out.println("starting batch " + (index + 1) + " of " + batches);
        toSarsa.executeBatch(RealScalar.of(0.1));
        DiscreteQsa toQsa = toSarsa.getQsa();
        XYtoSarsa.append(Tensors.vector(RealScalar.of(index).number(), errorAnalysis.getError(monteCarloInterface, optimalQsa, toQsa).number()));
      }
      System.out.println("Time for TrueOnlineSarsaTest: " + stopwatch.display_seconds() + "s");
      DiscreteQsa toQsa = toSarsa.getQsa();
      // System.out.println(toSarsa.getW());
      // toSarsa.printValues();
      // toSarsa.printPolicy();
      return XYtoSarsa;
    }
  },//
  ;
  public abstract Tensor analyse(MonteCarloInterface monteCarloInterface, int batches, DiscreteQsa optimalQsa, MonteCarloErrorAnalysis errorAnalysis);
}