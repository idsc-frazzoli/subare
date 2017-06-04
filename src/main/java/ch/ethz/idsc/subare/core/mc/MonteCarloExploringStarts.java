// code by jph
package ch.ethz.idsc.subare.core.mc;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import ch.ethz.idsc.subare.core.DiscreteModel;
import ch.ethz.idsc.subare.core.EpisodeDigest;
import ch.ethz.idsc.subare.core.EpisodeInterface;
import ch.ethz.idsc.subare.core.StepInterface;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.util.Average;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;
import ch.ethz.idsc.tensor.alg.Accumulate;
import ch.ethz.idsc.tensor.alg.Last;
import ch.ethz.idsc.tensor.alg.Multinomial;

/** Monte Carlo exploring starts improves an initial policy
 * based on average returns from complete episodes.
 * 
 * see box on p.107 */
public class MonteCarloExploringStarts implements EpisodeDigest {
  private final Scalar gamma;
  private final DiscreteQsa qsa;
  private final Map<Tensor, Average> map = new HashMap<>();

  /** @param discreteModel */
  public MonteCarloExploringStarts(DiscreteModel discreteModel) {
    this.gamma = discreteModel.gamma();
    this.qsa = DiscreteQsa.build(discreteModel); // <- "arbitrary"
  }

  @Override
  public void digest(EpisodeInterface episodeInterface) {
    Map<Tensor, Integer> first = new HashMap<>();
    Tensor rewards = Tensors.empty();
    List<StepInterface> trajectory = new ArrayList<>();
    while (episodeInterface.hasNext()) {
      StepInterface stepInterface = episodeInterface.step();
      Tensor key = DiscreteQsa.createKey(stepInterface.prevState(), stepInterface.action());
      if (!first.containsKey(key))
        first.put(key, trajectory.size());
      rewards.append(stepInterface.reward());
      trajectory.add(stepInterface);
    }
    Map<Tensor, Scalar> gains = new HashMap<>();
    if (gamma.equals(RealScalar.ONE)) {
      Tensor accumulate = Accumulate.of(rewards);
      for (Entry<Tensor, Integer> entry : first.entrySet()) {
        Tensor key = entry.getKey();
        Scalar alt = (Scalar) Last.of(accumulate);
        final int fromIndex = entry.getValue();
        if (0 < fromIndex)
          alt = alt.subtract(accumulate.Get(fromIndex - 1));
        // Scalar gain = Multinomial.horner(rewards.extract(fromIndex, rewards.length()), gamma);
        // if (!gain.equals(alt))
        // throw TensorRuntimeException.of(gain, alt);
        gains.put(key, alt);
      }
    } else {
      for (Entry<Tensor, Integer> entry : first.entrySet()) {
        Tensor key = entry.getKey();
        final int fromIndex = entry.getValue();
        Scalar gain = Multinomial.horner(rewards.extract(fromIndex, rewards.length()), gamma);
        gains.put(key, gain);
      }
    }
    // TODO more efficient update of average
    // compute average(Returns(s,a))
    for (StepInterface stepInterface : trajectory) {
      Tensor key = DiscreteQsa.createKey(stepInterface.prevState(), stepInterface.action());
      if (!map.containsKey(key))
        map.put(key, new Average());
      map.get(key).track(gains.get(key));
    }
    { // update
      for (Entry<Tensor, Average> entry : map.entrySet()) {
        Tensor key = entry.getKey();
        Tensor state = key.get(0);
        Tensor action = key.get(1);
        Scalar value = entry.getValue().get();
        // System.out.println(value);
        qsa.assign(state, action, value);
      }
    }
  }

  public DiscreteQsa qsa() {
    return qsa;
  }
}
