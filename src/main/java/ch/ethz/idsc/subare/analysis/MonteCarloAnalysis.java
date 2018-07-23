// code by fluric
package ch.ethz.idsc.subare.analysis;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import ch.ethz.idsc.subare.core.MonteCarloInterface;
import ch.ethz.idsc.subare.core.Policy;
import ch.ethz.idsc.subare.core.StandardModel;
import ch.ethz.idsc.subare.core.alg.ActionValueIterations;
import ch.ethz.idsc.subare.core.td.Sarsa;
import ch.ethz.idsc.subare.core.td.SarsaType;
import ch.ethz.idsc.subare.core.util.ConstantLearningRate;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.EGreedyPolicy;
import ch.ethz.idsc.subare.core.util.ExploringStarts;
import ch.ethz.idsc.subare.util.PlotUtils;
import ch.ethz.idsc.subare.util.Stopwatch;
import ch.ethz.idsc.tensor.DecimalScalar;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Tensor;

public enum MonteCarloAnalysis {
  ;
  public static void analyse(MonteCarloInterface monteCarloInterface, int trials, int batches, List<MonteCarloAlgorithms> algorithmList,
      List<DiscreteModelErrorAnalysis> errorAnalysisList) throws Exception {
    DiscreteQsa optimalQsa = getOptimalQsa(monteCarloInterface, batches);
    Map<String, Tensor> algorithmResults = new LinkedHashMap<>();
    // ---
    for (MonteCarloAlgorithms monteCarloAlgorithms : algorithmList)
      algorithmResults.put(monteCarloAlgorithms.name(),
          monteCarloAlgorithms.analyseNTimes(monteCarloInterface, batches, optimalQsa, errorAnalysisList, trials));
    PlotUtils.createPlot(algorithmResults,
        "Convergence_" + monteCarloInterface.getClass().getSimpleName().toString() + "_" + trials + "trials" + "_" + batches + "batches", errorAnalysisList);
  }

  public static DiscreteQsa getOptimalQsa(MonteCarloInterface monteCarloInterface, int batches) {
    if (!(monteCarloInterface instanceof StandardModel)) { // if no AVI is possible, try to approximate it
      System.out.println("Approximating optimal QSA because the model does not implement StandardModel!");
      DiscreteQsa qsaSarsa = DiscreteQsa.build(monteCarloInterface);
      final Sarsa sarsa = SarsaType.QLEARNING.supply(monteCarloInterface, qsaSarsa, ConstantLearningRate.of(RealScalar.of(0.05)));
      sarsa.setExplore(RealScalar.of(.1));
      Stopwatch stopwatch = Stopwatch.started();
      for (int index = 0; index < batches * 10; ++index) {
        Policy policy = EGreedyPolicy.bestEquiprobable(monteCarloInterface, sarsa.qsa(), RealScalar.of(.1));
        ExploringStarts.batch(monteCarloInterface, policy, 1, sarsa);
      }
      System.out.println("Time for optimal QSA approximation: " + stopwatch.display_seconds() + "s");
      return sarsa.qsa();
    }
    Stopwatch stopwatch = Stopwatch.started();
    DiscreteQsa optimalQsa = ActionValueIterations.solve((StandardModel) monteCarloInterface, DecimalScalar.of(.0001));
    System.out.println("Time for AVI: " + stopwatch.display_seconds() + "s");
    return optimalQsa;
  }

  public static void main(String[] args) throws Exception {
    MonteCarloInterface monteCarloInterface = MonteCarloExamples.AIRPORT.get();
    // ---
    List<MonteCarloAlgorithms> list = new ArrayList<>();
    // list.add(MonteCarloAlgorithms.MONTE_CARLO);
    // list.add(MonteCarloAlgorithms.ORIGINAL_SARSA);
    // list.add(MonteCarloAlgorithms.ORIGINAL_TRUE_ONLINE_SARSA);
    // list.add(MonteCarloAlgorithms.DOUBLE_ORIGINAL_SARSA);
    // list.add(MonteCarloAlgorithms.EXPECTED_SARSA);
    // list.add(MonteCarloAlgorithms.EXPECTED_TRUE_ONLINE_SARSA);
    // list.add(MonteCarloAlgorithms.DOUBLE_EXPECTED_SARSA);
    // list.add(MonteCarloAlgorithms.QLEARNING_SARSA);
    // list.add(MonteCarloAlgorithms.QLEARNING_TRUE_ONLINE_SARSA);
    // list.add(MonteCarloAlgorithms.DOUBLE_QLEARNING_SARSA);
    // ---
    List<DiscreteModelErrorAnalysis> errorAnalysis = new ArrayList<>();
    errorAnalysis.add(DiscreteModelErrorAnalysis.LINEAR_POLICY);
    errorAnalysis.add(DiscreteModelErrorAnalysis.LINEAR_QSA);
    // ---
    analyse(monteCarloInterface, 1, 10, list, errorAnalysis);
  }
}
