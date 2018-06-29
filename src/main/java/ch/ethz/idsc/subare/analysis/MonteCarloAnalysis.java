// code by fluric
package ch.ethz.idsc.subare.analysis;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import ch.ethz.idsc.subare.core.MonteCarloInterface;
import ch.ethz.idsc.subare.core.StandardModel;
import ch.ethz.idsc.subare.core.alg.ActionValueIterations;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
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

  public static void analyse(MonteCarloInterface mcInterface, int batches, List<MonteCarloAlgorithms> analysisAlgorithms) throws Exception {
    DiscreteQsa optimalQsa = getOptimalQsa(mcInterface);
    Map<String, Tensor> algorithmResults = new HashMap<>();
    // ---
    for (MonteCarloAlgorithms alg : analysisAlgorithms) {
      algorithmResults.put(alg.name(), alg.analyse(mcInterface, batches, optimalQsa));
    }
    PlotUtils.createPlot(algorithmResults, "Convergence_" + mcInterface.getClass().getSimpleName().toString());
  }

  public static DiscreteQsa getOptimalQsa(MonteCarloInterface mcInterface) {
    Stopwatch stopwatch = Stopwatch.started();
    DiscreteQsa optimalQsa = ActionValueIterations.solve((StandardModel) mcInterface, DecimalScalar.of(.0001));
    System.out.println("time for AVI: " + stopwatch.display_seconds() + "s");
    // DiscreteUtils.print(optimalQsa);
    // Policy policyQsa = GreedyPolicy.bestEquiprobable(mcInterface, optimalQsa);
    // Policies.print(policyQsa, airport.states());
    return optimalQsa;
  }

  public static void main(String[] args) throws Exception {
    MonteCarloInterface monteCarloInterface = AnalysisModels.WIRELOOP.supply();
    // ---
    List<MonteCarloAlgorithms> list = new ArrayList<>();
    list.add(MonteCarloAlgorithms.MonteCarlo);
    list.add(MonteCarloAlgorithms.OriginalSarsa);
    list.add(MonteCarloAlgorithms.ExpectedSarsa);
    list.add(MonteCarloAlgorithms.QLearningSarsa);
    list.add(MonteCarloAlgorithms.TrueOnlineSarsa);
    // ---
    analyse(monteCarloInterface, 10, list);
  }
}
