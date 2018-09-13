// code by fluric
package ch.ethz.idsc.subare.analysis;

import java.util.List;

import ch.ethz.idsc.subare.core.MonteCarloInterface;
import ch.ethz.idsc.subare.core.QsaInterface;
import ch.ethz.idsc.subare.core.StateActionCounter;
import ch.ethz.idsc.subare.core.td.SarsaType;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.DiscreteStateActionCounter;
import ch.ethz.idsc.subare.core.util.PolicyType;
import ch.ethz.idsc.subare.util.Stopwatch;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;
import ch.ethz.idsc.tensor.red.Mean;

public enum MonteCarloAlgorithms {
  ORIGINAL_SARSA() {
    @Override
    MonteCarloTrial create(MonteCarloInterface monteCarloInterface) {
      QsaInterface qsa = DiscreteQsa.build(monteCarloInterface);
      StateActionCounter sac = new DiscreteStateActionCounter();
      return SarsaMonteCarloTrial.of(monteCarloInterface, SarsaType.ORIGINAL, PolicyType.EGREEDY.bestEquiprobable(monteCarloInterface, qsa, sac));
    }
  }, //
  DOUBLE_ORIGINAL_SARSA() {
    @Override
    MonteCarloTrial create(MonteCarloInterface monteCarloInterface) {
      return DoubleSarsaMonteCarloTrial.of(monteCarloInterface, SarsaType.ORIGINAL);
    }
  }, //
  EXPECTED_SARSA() {
    @Override
    MonteCarloTrial create(MonteCarloInterface monteCarloInterface) {
      QsaInterface qsa = DiscreteQsa.build(monteCarloInterface);
      StateActionCounter sac = new DiscreteStateActionCounter();
      return SarsaMonteCarloTrial.of(monteCarloInterface, SarsaType.EXPECTED, PolicyType.EGREEDY.bestEquiprobable(monteCarloInterface, qsa, sac));
    }
  }, //
  DOUBLE_EXPECTED_SARSA() {
    @Override
    MonteCarloTrial create(MonteCarloInterface monteCarloInterface) {
      return DoubleSarsaMonteCarloTrial.of(monteCarloInterface, SarsaType.EXPECTED);
    }
  }, //
  QLEARNING_SARSA() {
    @Override
    MonteCarloTrial create(MonteCarloInterface monteCarloInterface) {
      QsaInterface qsa = DiscreteQsa.build(monteCarloInterface);
      StateActionCounter sac = new DiscreteStateActionCounter();
      return SarsaMonteCarloTrial.of(monteCarloInterface, SarsaType.QLEARNING, PolicyType.EGREEDY.bestEquiprobable(monteCarloInterface, qsa, sac));
    }
  }, //
  QLEARNING_SARSA_UCB() {
    @Override
    MonteCarloTrial create(MonteCarloInterface monteCarloInterface) {
      QsaInterface qsa = DiscreteQsa.build(monteCarloInterface);
      StateActionCounter sac = new DiscreteStateActionCounter();
      return SarsaMonteCarloTrial.of(monteCarloInterface, SarsaType.QLEARNING, PolicyType.UCB.bestEquiprobable(monteCarloInterface, qsa, sac));
    }
  }, //
  DOUBLE_QLEARNING_SARSA() {
    @Override
    MonteCarloTrial create(MonteCarloInterface monteCarloInterface) {
      return DoubleSarsaMonteCarloTrial.of(monteCarloInterface, SarsaType.QLEARNING);
    }
  }, //
  ORIGINAL_TRUE_ONLINE_SARSA() {
    @Override
    MonteCarloTrial create(MonteCarloInterface monteCarloInterface) {
      return TrueOnlineMonteCarloTrial.of(monteCarloInterface, SarsaType.ORIGINAL);
    }
  }, //
  EXPECTED_TRUE_ONLINE_SARSA() {
    @Override
    MonteCarloTrial create(MonteCarloInterface monteCarloInterface) {
      return TrueOnlineMonteCarloTrial.of(monteCarloInterface, SarsaType.EXPECTED);
    }
  }, //
  QLEARNING_TRUE_ONLINE_SARSA() {
    @Override
    MonteCarloTrial create(MonteCarloInterface monteCarloInterface) {
      return TrueOnlineMonteCarloTrial.of(monteCarloInterface, SarsaType.QLEARNING);
    }
  }, //
  MONTE_CARLO() {
    @Override
    MonteCarloTrial create(MonteCarloInterface monteCarloInterface) {
      QsaInterface qsa = DiscreteQsa.build(monteCarloInterface);
      StateActionCounter sac = new DiscreteStateActionCounter();
      return new EpisodeMonteCarloTrial(monteCarloInterface, PolicyType.EGREEDY.bestEquiprobable(monteCarloInterface, qsa, sac));
    }
  }, //
  ;
  abstract MonteCarloTrial create(MonteCarloInterface monteCarloInterface);

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
    MonteCarloTrial monteCarloTrial = create(monteCarloInterface);
    Tensor samples = Tensors.empty();
    for (int index = 0; index < batches; ++index) {
      // System.out.println("starting batch " + (index + 1) + " of " + batches);
      monteCarloTrial.executeBatch();
      Tensor vector = Tensors.vector(index);
      for (DiscreteModelErrorAnalysis errorAnalysis : errorAnalysisList)
        vector.append(errorAnalysis.getError(monteCarloInterface, optimalQsa, monteCarloTrial.qsa()));
      samples.append(vector);
    }
    return samples;
  }
}