from optparse import OptionParser
import json

def main():
    usage = "" # TODO 
    parser = OptionParser(usage=usage)
    #parser.add_option("-a", "--a_descrip", action="store_true", help="This is a flat")
    #parser.add_option("-b", "--b_descrip", help="This is an argument")
    (options, args) = parser.parse_args()

    with open("./tags.json", 'r') as f:
        j = json.load(f)
        source_to_targets = j['implications']        
        all_nodes = j['definitions']
    
    g = "digraph G {\n"
    for node in all_nodes:
        g += '"%s"\n' % node
    for source, targets in source_to_targets.iteritems():
        for target in targets:
            g += '"%s" -> "%s"\n' % (source, target)
    g += "}"


    print g
    

if __name__ == "__main__":
    main()
