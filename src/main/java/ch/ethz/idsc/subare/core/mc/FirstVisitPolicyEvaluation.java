// code by jph
package ch.ethz.idsc.subare.core.mc;

import ch.ethz.idsc.subare.core.EpisodeInterface;
import ch.ethz.idsc.subare.core.EpisodeSupplier;
import ch.ethz.idsc.subare.core.PolicyInterface;
import ch.ethz.idsc.subare.core.StandardModel;
import ch.ethz.idsc.subare.core.StepInterface;
import ch.ethz.idsc.subare.util.Index;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;
import ch.ethz.idsc.tensor.alg.Array;

public class FirstVisitPolicyEvaluation {
  final StandardModel standardModel;
  final PolicyInterface policyInterface; // TODO use policy
  Tensor returns;
  Tensor values; // TODO

  public FirstVisitPolicyEvaluation( //
      StandardModel standardModel, PolicyInterface policyInterface, EpisodeSupplier episodeSupplier) {
    this.standardModel = standardModel;
    this.policyInterface = policyInterface;
    Index statesIndex = Index.build(standardModel.states());
    returns = Array.of(l -> Tensors.empty(), statesIndex.size());
    values = Array.zeros(statesIndex.size());
    int iterations = 0;
    while (iterations < 100) {
      EpisodeInterface episodeInterface = episodeSupplier.kickoff();
      int steps = 0;
      while (episodeInterface.hasNext()) {
        Tensor state = episodeInterface.state();
        StepInterface stepInterface = episodeInterface.step();
        Tensor action = stepInterface.action();
        Scalar reward = stepInterface.reward();
        Tensor stateS = stepInterface.nextState();
        System.out.println(state + " + " + action + " == r=" + reward + " in " + stateS);
        ++steps;
      }
      System.out.println(steps);
      ++iterations;
    }
  }
}
