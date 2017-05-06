// code by jph
package ch.ethz.idsc.subare.core.mc;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import ch.ethz.idsc.subare.core.EpisodeInterface;
import ch.ethz.idsc.subare.core.EpisodeSupplier;
import ch.ethz.idsc.subare.core.PolicyInterface;
import ch.ethz.idsc.subare.core.StandardModel;
import ch.ethz.idsc.subare.core.StepInterface;
import ch.ethz.idsc.subare.util.Average;
import ch.ethz.idsc.subare.util.Index;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.alg.Array;

public class FirstVisitPolicyEvaluation {
  final StandardModel standardModel;
  final PolicyInterface policyInterface; // TODO use policy
  final EpisodeSupplier episodeSupplier;
  Map<Tensor, Average> map = new HashMap<>();

  public FirstVisitPolicyEvaluation( //
      StandardModel standardModel, PolicyInterface policyInterface, EpisodeSupplier episodeSupplier) {
    this.standardModel = standardModel;
    this.policyInterface = policyInterface;
    this.episodeSupplier = episodeSupplier;
  }

  public Tensor simulate(final int iterations) {
    int iteration = 0;
    while (iteration < iterations) {
      EpisodeInterface episodeInterface = episodeSupplier.kickoff(policyInterface);
      List<StepInterface> trajectory = new ArrayList<>();
      Map<Tensor, Integer> first = new HashMap<>();
      while (episodeInterface.hasNext()) {
        StepInterface stepInterface = episodeInterface.step();
        Tensor state = stepInterface.prevState();
        if (!first.containsKey(state))
          first.put(state, trajectory.size());
        trajectory.add(stepInterface);
      }
      // System.out.println(first);
      for (StepInterface stepInterface : trajectory) {
        Tensor stateP = stepInterface.prevState();
        Tensor action = stepInterface.action();
        Scalar reward = stepInterface.reward();
        Tensor stateS = stepInterface.nextState();
        Scalar G = RealScalar.ONE; // FIXME
        if (!map.containsKey(stateP))
          map.put(stateP, new Average());
        map.get(stateP).track(G);
        System.out.println(stateP + " + " + action + " == r=" + reward + " in " + stateS);
      }
      System.out.println(map);
      ++iteration;
    }
    Tensor states = standardModel.states();
    Index statesIndex = Index.build(states);
    Tensor values = Array.zeros(statesIndex.size());
    for (Entry<Tensor, Average> entry : map.entrySet())
      values.set(entry.getValue().get(), statesIndex.of(entry.getKey()));
    return values;
  }
}
