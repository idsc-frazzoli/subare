// code by jph
package ch.ethz.idsc.subare.core.mc;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import ch.ethz.idsc.subare.core.DiscreteModel;
import ch.ethz.idsc.subare.core.EpisodeInterface;
import ch.ethz.idsc.subare.core.EpisodeSupplier;
import ch.ethz.idsc.subare.core.PolicyInterface;
import ch.ethz.idsc.subare.core.StepInterface;
import ch.ethz.idsc.subare.core.util.DiscreteVs;
import ch.ethz.idsc.subare.util.Average;
import ch.ethz.idsc.subare.util.Index;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;
import ch.ethz.idsc.tensor.alg.Array;
import ch.ethz.idsc.tensor.alg.Multinomial;

/** see box on p.100 */
public class FirstVisitPolicyEvaluation {
  private final DiscreteModel standardModel;
  private final PolicyInterface policyInterface;
  private final Scalar gamma;
  private final EpisodeSupplier episodeSupplier;

  public FirstVisitPolicyEvaluation( //
      EpisodeSupplier episodeSupplier, PolicyInterface policyInterface, //
      DiscreteModel standardModel, Scalar gamma) {
    this.standardModel = standardModel;
    this.policyInterface = policyInterface;
    this.gamma = gamma;
    this.episodeSupplier = episodeSupplier;
  }

  public DiscreteVs simulate(final int iterations) {
    int iteration = 0;
    Map<Tensor, Average> map = new HashMap<>();
    while (iteration < iterations) {
      EpisodeInterface episodeInterface = episodeSupplier.kickoff(policyInterface);
      Map<Tensor, Integer> first = new HashMap<>();
      Map<Tensor, Scalar> gains = new HashMap<>();
      Tensor rewards = Tensors.empty();
      List<StepInterface> trajectory = new ArrayList<>();
      while (episodeInterface.hasNext()) {
        StepInterface stepInterface = episodeInterface.step();
        Tensor state = stepInterface.prevState();
        if (!first.containsKey(state))
          first.put(state, trajectory.size());
        rewards.append(stepInterface.reward());
        trajectory.add(stepInterface);
        // System.out.println(state+" "+stepInterface.action());
      }
      // System.out.println("reached final");
      for (Entry<Tensor, Integer> entry : first.entrySet()) {
        Tensor state = entry.getKey();
        int fromIndex = entry.getValue();
        gains.put(state, Multinomial.horner(rewards.extract(fromIndex, rewards.length()), gamma));
      }
      // TODO more efficient update of average
      for (StepInterface stepInterface : trajectory) {
        Tensor stateP = stepInterface.prevState();
        if (!map.containsKey(stateP))
          map.put(stateP, new Average());
        map.get(stateP).track(gains.get(stateP));
      }
      ++iteration;
    }
    Tensor states = standardModel.states();
    Index index = Index.build(states);
    Tensor values = Array.zeros(index.size());
    for (Entry<Tensor, Average> entry : map.entrySet())
      values.set(entry.getValue().get(), index.of(entry.getKey()));
    return new DiscreteVs(index, values);
  }
}
