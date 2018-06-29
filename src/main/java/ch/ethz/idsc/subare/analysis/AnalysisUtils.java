//code by fluric
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
import ch.ethz.idsc.tensor.DecimalScalar;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.red.Times;
import ch.ethz.idsc.tensor.sca.Abs;

public class AnalysisUtils {
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

  public static void analyse(MonteCarloInterface mcInterface, int batches, List<AnalysisAlgorithm> analysisAlgorithms) throws Exception {
    DiscreteQsa optimalQsa = getOptimalQsa(mcInterface);
    Map<String, Tensor> algorithmResults = new HashMap<>();
    // ---
    for (AnalysisAlgorithm alg : analysisAlgorithms) {
      algorithmResults.put(alg.getName(), alg.analyse(mcInterface, batches, optimalQsa));
    }
    PlotUtils.createPlot(algorithmResults, "Convergence_" + mcInterface.getClass().getSimpleName().toString());
  }

  public static DiscreteQsa getOptimalQsa(MonteCarloInterface mcInterface) {
    long start = System.currentTimeMillis();
    DiscreteQsa optimalQsa = ActionValueIterations.solve((StandardModel) mcInterface, DecimalScalar.of(.0001));
    System.out.println("time for AVI: " + (System.currentTimeMillis() - start) / 1000.0 + "s");
    // DiscreteUtils.print(optimalQsa);
    // Policy policyQsa = GreedyPolicy.bestEquiprobable(mcInterface, optimalQsa);
    // Policies.print(policyQsa, airport.states());
    return optimalQsa;
  }

  public static void main(String[] args) throws Exception {
    MonteCarloInterface mcInterface = AnalysisModels.WIRELOOP.supply();
    // ---
    List<AnalysisAlgorithm> algList = new ArrayList<>();
    algList.add(AnalysisAlgorithms.MONTECARLO.supply());
    algList.add(AnalysisAlgorithms.ORIGINALSARSA.supply());
    algList.add(AnalysisAlgorithms.EXPECTEDSARSA.supply());
    algList.add(AnalysisAlgorithms.QLEARNINGSARSA.supply());
    algList.add(AnalysisAlgorithms.TRUEONLINESARSA.supply());
    // ---
    analyse(mcInterface, 10, algList);
  }
}
