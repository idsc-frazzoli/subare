// code by fluric
package ch.ethz.idsc.subare.util;

import java.util.Random;

import ch.ethz.idsc.tensor.RationalScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.sca.Clips;

public class Coinflip {
  /** @param p_head in the interval [0, 1]
   * @return new instance of Coinflip with given probability p_head that {@link #tossHead()} returns true
   * @throws Exception if given probability is not inside the unit interval */
  public static Coinflip of(Scalar p_head) {
    return new Coinflip(Clips.unit().requireInside(p_head));
  }

  /** Quote from Wikipedia:
   * "a fair coin is an idealized randomizing device with two states
   * (usually named "heads" and "tails") which are equally likely to occur."
   * 
   * @return new instance of Coinflip for which {@link #tossHead()} returns true with probability 1/2 */
  public static Coinflip fair() {
    return new Coinflip(RationalScalar.HALF);
  }

  // ---
  private final Random random = new Random();
  private final float p_head;

  private Coinflip(Scalar p_head) {
    this.p_head = p_head.number().floatValue();
  }

  /** returns true if the coin toss ended up with head */
  public boolean tossHead() {
    return random.nextFloat() < p_head;
  }
}
