// code by jph
package ch.ethz.idsc.subare.core.mc;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import ch.ethz.idsc.subare.core.DiscreteModel;
import ch.ethz.idsc.subare.core.EpisodeInterface;
import ch.ethz.idsc.subare.core.EpisodeVsEstimator;
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
public class FirstVisitPolicyEvaluation implements EpisodeVsEstimator {
  private final DiscreteModel discreteModel;
  private final Scalar gamma;
  final DiscreteVs vs;
  final Map<Tensor, Average> map = new HashMap<>(); // TODO no good!

  public FirstVisitPolicyEvaluation(DiscreteModel discreteModel, DiscreteVs vs) {
    this.discreteModel = discreteModel;
    this.gamma = discreteModel.gamma();
    this.vs = vs; // TODO write results directly in vs!
  }

  @Override
  public void digest(EpisodeInterface episodeInterface) {
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
  }

  @Override
  public DiscreteVs vs() {
    Tensor states = discreteModel.states();
    Index index = Index.build(states);
    Tensor values = Array.zeros(index.size());
    for (Entry<Tensor, Average> entry : map.entrySet())
      values.set(entry.getValue().Get(), index.of(entry.getKey()));
    return new DiscreteVs(index, values);
  }
}
