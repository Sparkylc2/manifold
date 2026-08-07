


- add colour bar to the cylinder and fea flutter


## Rocket landing
- fix colours for rocket landing (both bg and thrust)
- make the PID controller look more compact and less thrown together
- move it down a bit so we can have text at the top
- fix the pid tuning so it lands a little faster without that oscillation
- have it use a far simpler drawing for both body and thruster
- fix clipping of thruster into ground
- make the bend faster (but not so fast it immediately goes to max extent)
- increase colour bar size for temp and stress so that they are actually visible



# 1.
## jansen
- done
## cart kickup
- done
## engine
- looks good, done

# 2
## fea flutter
- fix the amount but lwkey lets just leave it at that. we can fuck with it once we slow down speed
- change to "fluid\n speed" and "stress" on the bottom
## foil flutter
- looks good, just sizing
## nozzle
- definitely in the video have it sped up a bit
- draw the nozzle procedurally, looks weird right now

# 3
## pod 
- do we want the bars to only occupy ~half if at 46% KE? right now they maximize
- outside of that, good, colours are ice, idea is pretty polished off. maybe fix the sizing of the 
bar so they align with the size of the box aorund the mode (which its slightly larger right now)

## esn + lstm
- we will have both fully trained and fitted properly, and show a live reconstruction going forward however many seconds, as well as a reconstruction error plot
- will run next to eachother at the same time, just do what the current esn demo does


# 4
wave pde plot, maybe a vector field plot with attractors or some other weird math object but rendered beautifully, and then maybe something else
 

