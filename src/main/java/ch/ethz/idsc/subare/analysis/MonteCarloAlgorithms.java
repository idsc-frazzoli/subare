// code by fluric
package ch.ethz.idsc.subare.analysis;

import java.util.List;

import ch.ethz.idsc.subare.core.DiscreteQsaSupplier;
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
import ch.ethz.idsc.subare.core.util.EGreedyPolicy;
import ch.ethz.idsc.subare.core.util.ExactFeatureMapper;
import ch.ethz.idsc.subare.core.util.ExploringStarts;
import ch.ethz.idsc.subare.core.util.FeatureMapper;
import ch.ethz.idsc.subare.util.GlobalAssert;
import ch.ethz.idsc.subare.util.Stopwatch;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;
import ch.ethz.idsc.tensor.red.Mean;

public enum MonteCarloAlgorithms {
  OriginalSarsa() {
    @Override
    public DiscreteQsaSupplier getAlgorithm(MonteCarloInterface monteCarloInterface) {
      DiscreteQsa qsaSarsa = DiscreteQsa.build(monteCarloInterface);
      LearningRate learningRate = ConstantLearningRate.of(RealScalar.of(0.05));
      final Sarsa sarsa = new OriginalSarsa(monteCarloInterface, qsaSarsa, learningRate);
      sarsa.setExplore(RealScalar.of(0.1));
      return sarsa;
    }

    @Override
    public void executeBatch(DiscreteQsaSupplier algorithm, MonteCarloInterface monteCarloInterface) {
      GlobalAssert.that(algorithm instanceof OriginalSarsa);
      Policy policy = EGreedyPolicy.bestEquiprobable(monteCarloInterface, algorithm.qsa(), RealScalar.of(0.1));
      ExploringStarts.batch(monteCarloInterface, policy, 1, (OriginalSarsa) algorithm);
    }
  }, //
  ExpectedSarsa() {
    @Override
    public DiscreteQsaSupplier getAlgorithm(MonteCarloInterface monteCarloInterface) {
      DiscreteQsa qsaSarsa = DiscreteQsa.build(monteCarloInterface);
      LearningRate learningRate = ConstantLearningRate.of(RealScalar.of(0.05));
      final Sarsa sarsa = new ExpectedSarsa(monteCarloInterface, qsaSarsa, learningRate);
      sarsa.setExplore(RealScalar.of(0.1));
      return sarsa;
    }

    @Override
    public void executeBatch(DiscreteQsaSupplier algorithm, MonteCarloInterface monteCarloInterface) {
      GlobalAssert.that(algorithm instanceof ExpectedSarsa);
      Policy policy = EGreedyPolicy.bestEquiprobable(monteCarloInterface, algorithm.qsa(), RealScalar.of(.1));
      ExploringStarts.batch(monteCarloInterface, policy, 1, (ExpectedSarsa) algorithm);
    }
  }, //
  QLearningSarsa() {
    @Override
    public DiscreteQsaSupplier getAlgorithm(MonteCarloInterface monteCarloInterface) {
      DiscreteQsa qsaSarsa = DiscreteQsa.build(monteCarloInterface);
      LearningRate learningRate = ConstantLearningRate.of(RealScalar.of(0.05));
      final Sarsa sarsa = new QLearning(monteCarloInterface, qsaSarsa, learningRate);
      sarsa.setExplore(RealScalar.of(0.1));
      return sarsa;
    }

    @Override
    public void executeBatch(DiscreteQsaSupplier algorithm, MonteCarloInterface monteCarloInterface) {
      GlobalAssert.that(algorithm instanceof QLearning);
      Policy policy = EGreedyPolicy.bestEquiprobable(monteCarloInterface, algorithm.qsa(), RealScalar.of(.1));
      ExploringStarts.batch(monteCarloInterface, policy, 1, (QLearning) algorithm);
    }
  }, //
  DoubleQLearningSarsa() {
    @Override
    public DiscreteQsaSupplier getAlgorithm(MonteCarloInterface monteCarloInterface) {
      DiscreteQsa qsa1 = DiscreteQsa.build(monteCarloInterface);
      DiscreteQsa qsa2 = DiscreteQsa.build(monteCarloInterface);
      DoubleSarsa sarsa = new DoubleSarsa(SarsaType.QLEARNING, monteCarloInterface, //
          qsa1, qsa2, //
          DefaultLearningRate.of(1, .51), //
          DefaultLearningRate.of(1, .51));
      sarsa.setExplore(RealScalar.of(0.1));
      return sarsa;
    }

    @Override
    public void executeBatch(DiscreteQsaSupplier algorithm, MonteCarloInterface monteCarloInterface) {
      GlobalAssert.that(algorithm instanceof DoubleSarsa);
      Policy policy = ((DoubleSarsa) algorithm).getEGreedy();
      ExploringStarts.batch(monteCarloInterface, policy, 1, (DoubleSarsa) algorithm);
    }
  }, //
  MonteCarlo() {
    @Override
    public DiscreteQsaSupplier getAlgorithm(MonteCarloInterface monteCarloInterface) {
      return new MonteCarloExploringStarts(monteCarloInterface);
    }

    @Override
    public void executeBatch(DiscreteQsaSupplier algorithm, MonteCarloInterface monteCarloInterface) {
      GlobalAssert.that(algorithm instanceof MonteCarloExploringStarts);
      Policy policyMC = EGreedyPolicy.bestEquiprobable(monteCarloInterface, algorithm.qsa(), RealScalar.of(.1));
      ExploringStarts.batch(monteCarloInterface, policyMC, (MonteCarloExploringStarts) algorithm);
    }
  }, //
  TrueOnlineSarsa() {
    @Override
    public DiscreteQsaSupplier getAlgorithm(MonteCarloInterface monteCarloInterface) {
      FeatureMapper featureMapper = new ExactFeatureMapper(monteCarloInterface);
      LearningRate learningRate = ConstantLearningRate.of(RealScalar.of(0.05));
      return new TrueOnlineSarsa(monteCarloInterface, RealScalar.of(0.7), learningRate, featureMapper);
    }

    @Override
    public void executeBatch(DiscreteQsaSupplier algorithm, MonteCarloInterface monteCarloInterface) {
      GlobalAssert.that(algorithm instanceof TrueOnlineSarsa);
      ExploringStarts.batch(RealScalar.of(0.1), monteCarloInterface, (TrueOnlineSarsa) algorithm);
    }
  }, //
  TrueOnlineSarsaColdStart() {
    @Override
    public DiscreteQsaSupplier getAlgorithm(MonteCarloInterface monteCarloInterface) {
      FeatureMapper featureMapper = new ExactFeatureMapper(monteCarloInterface);
      LearningRate learningRate = ConstantLearningRate.of(RealScalar.of(0.2), false);
      return new TrueOnlineSarsa(monteCarloInterface, RealScalar.of(0.7), learningRate, featureMapper);
    }

    @Override
    public void executeBatch(DiscreteQsaSupplier algorithm, MonteCarloInterface monteCarloInterface) {
      GlobalAssert.that(algorithm instanceof TrueOnlineSarsa);
      ExploringStarts.batch(RealScalar.of(0.1), monteCarloInterface, (TrueOnlineSarsa) algorithm);
    }
  }, //
  TrueOnlineSarsaZero() {
    @Override
    public DiscreteQsaSupplier getAlgorithm(MonteCarloInterface monteCarloInterface) {
      FeatureMapper featureMapper = new ExactFeatureMapper(monteCarloInterface);
      LearningRate learningRate = ConstantLearningRate.of(RealScalar.of(0.05));
      return new TrueOnlineSarsa(monteCarloInterface, RealScalar.of(0.0), learningRate, featureMapper);
    }

    @Override
    public void executeBatch(DiscreteQsaSupplier algorithm, MonteCarloInterface monteCarloInterface) {
      GlobalAssert.that(algorithm instanceof TrueOnlineSarsa);
      ExploringStarts.batch(RealScalar.of(0.1), monteCarloInterface, (TrueOnlineSarsa) algorithm);
    }
  }, //
  TrueOnlineSarsaTest() {
    @Override
    public DiscreteQsaSupplier getAlgorithm(MonteCarloInterface monteCarloInterface) {
      FeatureMapper featureMapper = new ExactFeatureMapper(monteCarloInterface);
      LearningRate learningRate = ConstantLearningRate.of(RealScalar.of(0.05));
      return new TrueOnlineSarsa(monteCarloInterface, RealScalar.of(0.0), learningRate, featureMapper);
    }

    @Override
    public void executeBatch(DiscreteQsaSupplier algorithm, MonteCarloInterface monteCarloInterface) {
      GlobalAssert.that(algorithm instanceof TrueOnlineSarsa);
      ExploringStarts.batch(RealScalar.of(0.1), monteCarloInterface, (TrueOnlineSarsa) algorithm);
    }
  },//
  ;
  public abstract DiscreteQsaSupplier getAlgorithm(MonteCarloInterface monteCarloInterface);

  public abstract void executeBatch(DiscreteQsaSupplier algorithm, MonteCarloInterface monteCarloInterface);
  // public abstract Tensor analyse(MonteCarloInterface monteCarloInterface, int batches, DiscreteQsa optimalQsa, List<MonteCarloErrorAnalysis> errorAnalysis);

  public Tensor analyseNTimes(MonteCarloInterface monteCarloInterface, int batches, DiscreteQsa optimalQsa, List<DiscreteModelErrorAnalysis> errorAnalysis,
      int nTimes) {
    Tensor nSamples = Tensors.empty();
    Stopwatch stopwatch = Stopwatch.started();
    Stopwatch subWatch = Stopwatch.started();
    for (int i = 0; i < nTimes; ++i) {
      nSamples.append(analyseAlgorithm(monteCarloInterface, batches, optimalQsa, errorAnalysis));
      if (subWatch.display_seconds() > 10.0) {
        System.out.println(this.name() + " has finished trial " + i);
        subWatch = Stopwatch.started();
      }
    }
    System.out.println("Time for executing " + this.name() + " " + nTimes + " times with " + batches + " batches: " + stopwatch.display_seconds() + "s");
    return Mean.of(nSamples);
  }

  private Tensor analyseAlgorithm(MonteCarloInterface monteCarloInterface, int batches, DiscreteQsa optimalQsa,
      List<DiscreteModelErrorAnalysis> errorAnalysisList) {
    Tensor samples = Tensors.empty();
    DiscreteQsaSupplier algorithm = getAlgorithm(monteCarloInterface);
    for (int index = 0; index < batches; ++index) {
      Tensor sample = Tensors.vector(index);
      // System.out.println("starting batch " + (index + 1) + " of " + batches);
      executeBatch(algorithm, monteCarloInterface);
      for (DiscreteModelErrorAnalysis errorAnalysis : errorAnalysisList) {
        sample.append(errorAnalysis.getError(monteCarloInterface, optimalQsa, algorithm.qsa()));
      }
      samples.append(sample);
    }
    return samples;
  }
}