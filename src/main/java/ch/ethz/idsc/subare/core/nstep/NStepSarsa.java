// code by  jph
package ch.ethz.idsc.subare.core.nstep;

import java.util.LinkedList;

import ch.ethz.idsc.subare.core.EpisodeInterface;
import ch.ethz.idsc.subare.core.EpisodeSupplier;
import ch.ethz.idsc.subare.core.PolicyInterface;
import ch.ethz.idsc.subare.core.QsaInterface;
import ch.ethz.idsc.subare.core.StepInterface;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.alg.Multinomial;

/** n-step Sarsa for estimating Q(s,a)
 * 
 * box on p. 157 */
// TODO not tested yet
public class NStepSarsa {
  private final EpisodeSupplier episodeSupplier;
  private final PolicyInterface policyInterface;
  private final QsaInterface qsa;
  private final Scalar gamma;
  private final Scalar alpha;
  private final int size;

  public NStepSarsa( //
      EpisodeSupplier episodeSupplier, PolicyInterface policyInterface, //
      QsaInterface qsa, Scalar gamma, Scalar alpha, int size) {
    this.episodeSupplier = episodeSupplier;
    this.policyInterface = policyInterface;
    this.qsa = qsa;
    this.gamma = gamma;
    this.alpha = alpha;
    this.size = size;
  }

  public void simulate() {
    // TODO bound step count (to also support infinite episodes)
    EpisodeInterface episodeInterface = episodeSupplier.kickoff(policyInterface);
    LinkedList<StepInterface> list = new LinkedList<>();
    while (episodeInterface.hasNext()) {
      StepInterface stepInterface = episodeInterface.step();
      list.add(stepInterface);
      if (size == list.size()) {
        StepInterface last = list.getLast();
        Tensor rewards = Tensor.of(list.stream().map(StepInterface::reward));
        rewards.append(qsa.value(last.prevState(), last.action())); // TODO changed from next to prev!
        Scalar G = Multinomial.horner(rewards, gamma);
        StepInterface first = list.getFirst();
        Scalar value = qsa.value(first.prevState(), first.action());
        qsa.assign(first.prevState(), first.action(), value.add(G.subtract(value).multiply(alpha)));
        list.removeFirst();
      }
    }
    while (!list.isEmpty()) {
      Tensor rewards = Tensor.of(list.stream().map(StepInterface::reward));
      Scalar G = Multinomial.horner(rewards, gamma);
      StepInterface first = list.getFirst();
      Scalar value = qsa.value(first.prevState(), first.action());
      qsa.assign(first.prevState(), first.action(), value.add(G.subtract(value).multiply(alpha)));
      list.removeFirst();
    }
    // TODO ensure that policy is epsilon greedy
  }
}
