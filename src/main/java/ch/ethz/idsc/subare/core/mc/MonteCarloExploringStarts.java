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
import ch.ethz.idsc.subare.core.EpisodeSupplier;
import ch.ethz.idsc.subare.core.PolicyInterface;
import ch.ethz.idsc.subare.core.StepInterface;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.EGreedyPolicy;
import ch.ethz.idsc.subare.util.Average;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;
import ch.ethz.idsc.tensor.alg.Multinomial;

/** Monte Carlo exploring starts improves an initial policy
 * based on average returns from complete episodes.
 * 
 * see box on p.107 */
public class MonteCarloExploringStarts implements EpisodeDigest {
  private final EpisodeSupplier episodeSupplier;
  private PolicyInterface policy; // <- changes over the course of the iterations
  private final DiscreteModel discreteModel;
  private final Scalar gamma;
  private Scalar epsilon; // probability of exploration
  private final DiscreteQsa qsa;
  private final Map<Tensor, Average> map = new HashMap<>();

  /** @param episodeSupplier
   * @param policyInterface
   * @param discreteModel
   * @param gamma
   * @param epsilon */
  public MonteCarloExploringStarts( //
      EpisodeSupplier episodeSupplier, PolicyInterface policyInterface, //
      DiscreteModel discreteModel, Scalar gamma, Scalar epsilon) {
    // TODO check exploring starts
    this.episodeSupplier = episodeSupplier;
    this.policy = policyInterface;
    this.discreteModel = discreteModel;
    this.gamma = gamma;
    this.epsilon = epsilon;
    this.qsa = DiscreteQsa.build(discreteModel); // <- "arbitrary"
  }

  public void simulate(final int iterations) {
    int iteration = 0;
    while (iteration < iterations) {
      step();
      ++iteration;
    }
  }

  public void setExplorationProbability(Scalar epsilon) {
    this.epsilon = epsilon;
  }

  public void step() {
    // policy has to satisfy exploring starts condition
    EpisodeInterface episodeInterface = episodeSupplier.kickoff(policy);
    digest(episodeInterface);
  }

  public DiscreteQsa qsa() {
    return qsa;
  }

  @Override
  public void digest(EpisodeInterface episodeInterface) {
    Map<Tensor, Integer> first = new HashMap<>();
    Map<Tensor, Scalar> gains = new HashMap<>();
    Tensor rewards = Tensors.empty();
    List<StepInterface> trajectory = new ArrayList<>();
    while (episodeInterface.hasNext()) {
      StepInterface stepInterface = episodeInterface.step();
      Tensor key = DiscreteQsa.createKey(stepInterface.prevState(), stepInterface.action());
      if (!first.containsKey(key))
        first.put(key, trajectory.size());
      rewards.append(stepInterface.reward());
      trajectory.add(stepInterface);
      // System.out.println(state+" "+stepInterface.action());
    }
    // System.out.println("reached final");
    for (Entry<Tensor, Integer> entry : first.entrySet()) {
      Tensor key = entry.getKey();
      int fromIndex = entry.getValue();
      gains.put(key, Multinomial.horner(rewards.extract(fromIndex, rewards.length()), gamma));
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
        qsa.assign(state, action, entry.getValue().get());
      }
      policy = EGreedyPolicy.bestEquiprobable(discreteModel, qsa, epsilon);
    }
  }
}
