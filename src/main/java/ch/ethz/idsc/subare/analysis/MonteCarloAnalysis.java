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
import ch.ethz.idsc.subare.core.td.QLearning;
import ch.ethz.idsc.subare.core.td.Sarsa;
import ch.ethz.idsc.subare.core.util.ConstantLearningRate;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.EGreedyPolicy;
import ch.ethz.idsc.subare.core.util.ExploringStarts;
import ch.ethz.idsc.subare.util.GlobalAssert;
import ch.ethz.idsc.subare.util.PlotUtils;
import ch.ethz.idsc.subare.util.Stopwatch;
import ch.ethz.idsc.tensor.DecimalScalar;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.red.Times;
import ch.ethz.idsc.tensor.sca.Abs;

public enum MonteCarloAnalysis {
  ;
  public static Scalar getLinearQsaError(DiscreteQsa refQsa, DiscreteQsa currentQsa) {
    GlobalAssert.that(refQsa.size() == currentQsa.size());
    Scalar error = RealScalar.ZERO;
    Scalar delta;
    for (int index = 0; index < refQsa.size(); ++index) {
      delta = Abs.of(refQsa.values().get(index).subtract(currentQsa.values().get(index))).Get();
      error = error.add(delta);
    }
    return error;
  }

  public static Scalar getSquareQsaError(DiscreteQsa refQsa, DiscreteQsa currentQsa) {
    GlobalAssert.that(refQsa.size() == currentQsa.size());
    Scalar error = RealScalar.ZERO;
    Scalar delta;
    for (int index = 0; index < refQsa.size(); ++index) {
      delta = Abs.of(refQsa.values().get(index).subtract(currentQsa.values().get(index))).Get();
      error = error.add(Times.of(delta, delta));
    }
    return error;
  }

  public static void analyse(MonteCarloInterface monteCarloInterface, int batches, List<MonteCarloAlgorithms> list) throws Exception {
    DiscreteQsa optimalQsa = getOptimalQsa(monteCarloInterface, batches);
    Map<String, Tensor> algorithmResults = new LinkedHashMap<>();
    // ---
    for (MonteCarloAlgorithms monteCarloAlgorithms : list)
      algorithmResults.put(monteCarloAlgorithms.name(), monteCarloAlgorithms.analyse(monteCarloInterface, batches, optimalQsa));
    PlotUtils.createPlot(algorithmResults, "Convergence_" + monteCarloInterface.getClass().getSimpleName().toString());
    // Policies.print(GreedyPolicy.bestEquiprobable(monteCarloInterface, optimalQsa), monteCarloInterface.states());
  }

  public static DiscreteQsa getOptimalQsa(MonteCarloInterface monteCarloInterface, int batches) {
    if (!(monteCarloInterface instanceof StandardModel)) { // if no AVI is possible, try to approximate it
      System.out.println("Approximating optimal QSA because the model does not implement StandardModel!");
      DiscreteQsa qsaSarsa = DiscreteQsa.build(monteCarloInterface);
      final Sarsa sarsa = new QLearning(monteCarloInterface, qsaSarsa, ConstantLearningRate.of(RealScalar.of(0.05)));
      sarsa.setExplore(RealScalar.of(.1));
      Stopwatch stopwatch = Stopwatch.started();
      for (int index = 0; index < batches * 10; ++index) {
        // System.out.println("starting batch " + (index + 1) + " of " + batches);
        Policy policy = EGreedyPolicy.bestEquiprobable(monteCarloInterface, sarsa.qsa(), RealScalar.of(.1));
        ExploringStarts.batch(monteCarloInterface, policy, 1, sarsa);
      }
      // DiscreteUtils.print(sarsa.qsa());
      // Policies.print(GreedyPolicy.bestEquiprobable(monteCarloInterface, sarsa.qsa()), monteCarloInterface.states());
      return sarsa.qsa();
    }
    Stopwatch stopwatch = Stopwatch.started();
    DiscreteQsa optimalQsa = ActionValueIterations.solve((StandardModel) monteCarloInterface, DecimalScalar.of(.0001));
    System.out.println("time for AVI: " + stopwatch.display_seconds() + "s");
    // DiscreteUtils.print(optimalQsa);
    // Policy policyQsa = GreedyPolicy.bestEquiprobable(monteCarloInterface, optimalQsa);
    // Policies.print(policyQsa, airport.states());
    return optimalQsa;
  }

  public static void main(String[] args) throws Exception {
    MonteCarloInterface monteCarloInterface = MonteCarloExamples.VIRTUALSTATIONS.get();
    // ---
    List<MonteCarloAlgorithms> list = new ArrayList<>();
    // list.add(MonteCarloAlgorithms.MonteCarlo);
    list.add(MonteCarloAlgorithms.OriginalSarsa);
    // list.add(MonteCarloAlgorithms.ExpectedSarsa);
    list.add(MonteCarloAlgorithms.QLearningSarsa);
    list.add(MonteCarloAlgorithms.TrueOnlineSarsaMod);
    list.add(MonteCarloAlgorithms.TrueOnlineSarsa);
    // ---
    analyse(monteCarloInterface, 10, list);
  }
}
